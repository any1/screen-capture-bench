/*
 * Screen capturing benchmark for Linux.
 *
 * Renders a rotating triangle with Vulkan into a DRM-modifier image, copies the
 * rendered frame to a host (malloc'd) buffer, and optionally encodes the frames
 * to an MJPEG file for verification.  Every combination of GPU, buffer modifier,
 * shadow-buffer usage and mapping method (GBM vs. Vulkan) is benchmarked.
 */

#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <getopt.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/utsname.h>
#include <sys/sysmacros.h>
#include <sys/types.h>

#include <vulkan/vulkan.h>
#include <gbm.h>
#include <xf86drm.h>
#include <drm_fourcc.h>
#include <turbojpeg.h>
#include <json-c/json.h>

#include "triangle_vert.h"
#include "triangle_frag.h"

/* The single pixel format used throughout the benchmark.  VK_FORMAT_B8G8R8A8 in
 * memory is byte order B, G, R, A which matches DRM_FORMAT_ARGB8888. */
#define VK_FMT		VK_FORMAT_B8G8R8A8_UNORM
#define DRM_FMT		DRM_FORMAT_ARGB8888
#define BYTES_PER_PIXEL	4

/* Keep the memcpy honest: the copy must never be optimised away. */
static volatile uint64_t g_sink;

enum mapping_method {
	MAP_GBM,
	MAP_VULKAN,
};

enum shadow_mode {
	SHADOW_NONE,
	SHADOW_USED,
};

/* The fastest non-skipped combination seen so far. */
struct best {
	bool valid;
	double fps;
	char gpu[VK_MAX_PHYSICAL_DEVICE_NAME_SIZE];
	char modifier[128];
	enum mapping_method map;
	enum shadow_mode shadow;
};

struct options {
	double duration;	/* seconds */
	uint32_t width;
	uint32_t height;
	bool encode;		/* write MJPEG output for verification */
	const char *out_prefix;
};

struct gpu {
	VkPhysicalDevice phys;
	VkDevice dev;
	VkQueue queue;
	uint32_t qfamily;
	char name[VK_MAX_PHYSICAL_DEVICE_NAME_SIZE];
	VkPhysicalDeviceMemoryProperties memprops;

	/* GPU timestamp support, for measuring on-GPU time per frame. */
	float timestamp_period;		/* nanoseconds per timestamp tick */
	uint32_t timestamp_valid_bits;	/* 0 if timestamps are unsupported */

	int drm_fd;
	struct gbm_device *gbm;

	/* Re-used rendering resources (depend only on the format / size). */
	VkShaderModule vert;
	VkShaderModule frag;
	VkPipelineLayout playout;
	VkRenderPass rpass_ref;	/* compatible template for the pipeline */
	VkPipeline pipeline;
	VkCommandPool cmdpool;

	PFN_vkGetMemoryFdKHR GetMemoryFdKHR;
	PFN_vkGetImageDrmFormatModifierPropertiesEXT GetModifierProps;
};

/* Everything a single test combination is run against, except the mapping and
 * shadow choice that distinguish the combinations themselves. */
struct test_context {
	struct gpu *gpu;
	const struct options *opt;
	uint64_t modifier;
	const char *modname;
	bool optimal;		/* the implementation's optimal-tiling modifier */
	struct best *best;	/* updated with the fastest combination seen */
};

/* Per-test rendering target and capture resources. */
struct target {
	VkImage image;
	VkDeviceMemory mem;
	bool mem_coherent;
	VkImageView view;
	VkRenderPass rpass;
	VkFramebuffer fb;
	VkCommandBuffer cmd;
	VkFence fence;
	VkQueryPool qpool;	/* two timestamps: start/end of the frame */

	VkSubresourceLayout layout;	/* plane 0 of the image */

	/* Vulkan mapping (no shadow): persistent map of the image memory. */
	void *image_map;

	/* Shadow buffer (Vulkan mapping with shadow). */
	VkBuffer shadow_buf;
	VkDeviceMemory shadow_mem;
	bool shadow_coherent;
	void *shadow_map;

	/* GBM mapping. */
	int dmabuf_fd;
	struct gbm_bo *bo;
};

static double now_sec(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* Process-wide consumed CPU time (user + system), for cpu-per-frame. */
static double now_cpu_sec(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
	return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static int find_mem_type(const VkPhysicalDeviceMemoryProperties *mp,
			 uint32_t type_bits, VkMemoryPropertyFlags required,
			 VkMemoryPropertyFlags preferred)
{
	/* First try to satisfy the preferred flags too, then fall back. */
	for (int pass = 0; pass < 2; pass++) {
		VkMemoryPropertyFlags want = required | (pass == 0 ? preferred : 0);
		for (uint32_t i = 0; i < mp->memoryTypeCount; i++) {
			if (!(type_bits & (1u << i)))
				continue;
			if ((mp->memoryTypes[i].propertyFlags & want) == want)
				return (int)i;
		}
	}
	return -1;
}

static const char *modifier_name(uint64_t mod, char *buf, size_t len)
{
	char *name = drmGetFormatModifierName(mod);
	if (name) {
		snprintf(buf, len, "%s", name);
		free(name);
	} else {
		snprintf(buf, len, "0x%016" PRIx64, mod);
	}
	return buf;
}

/*
 * Vulkan instance / device setup.
 */

static const char *const k_dev_exts[] = {
	VK_EXT_IMAGE_DRM_FORMAT_MODIFIER_EXTENSION_NAME,
	VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
	VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME,
	VK_KHR_IMAGE_FORMAT_LIST_EXTENSION_NAME,
	VK_EXT_PHYSICAL_DEVICE_DRM_EXTENSION_NAME,
};

static bool dev_supports(VkPhysicalDevice phys, const char *ext)
{
	uint32_t n = 0;
	vkEnumerateDeviceExtensionProperties(phys, NULL, &n, NULL);
	VkExtensionProperties *props = calloc(n, sizeof(*props));
	vkEnumerateDeviceExtensionProperties(phys, NULL, &n, props);
	bool found = false;
	for (uint32_t i = 0; i < n; i++) {
		if (strcmp(props[i].extensionName, ext) == 0) {
			found = true;
			break;
		}
	}
	free(props);
	return found;
}

static int open_drm_render_node(VkPhysicalDevice phys)
{
	VkPhysicalDeviceDrmPropertiesEXT drmp = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRM_PROPERTIES_EXT,
	};
	VkPhysicalDeviceProperties2 p2 = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
		.pNext = &drmp,
	};

	if (dev_supports(phys, VK_EXT_PHYSICAL_DEVICE_DRM_EXTENSION_NAME)) {
		vkGetPhysicalDeviceProperties2(phys, &p2);
		if (drmp.hasRender) {
			dev_t want = makedev(drmp.renderMajor, drmp.renderMinor);
			char path[64];
			for (int i = 128; i < 192; i++) {
				snprintf(path, sizeof(path),
					 "/dev/dri/renderD%d", i);
				struct stat st;
				if (stat(path, &st) == 0 && st.st_rdev == want)
					return open(path, O_RDWR | O_CLOEXEC);
			}
		}
	}

	/* Fallback: just grab the first render node. */
	return open("/dev/dri/renderD128", O_RDWR | O_CLOEXEC);
}

static VkShaderModule make_shader(VkDevice dev, const uint32_t *code, size_t len)
{
	VkShaderModuleCreateInfo ci = {
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = len,
		.pCode = code,
	};
	VkShaderModule m = VK_NULL_HANDLE;
	vkCreateShaderModule(dev, &ci, NULL, &m);
	return m;
}

static VkRenderPass make_render_pass(VkDevice dev, VkImageLayout final_layout)
{
	VkAttachmentDescription att = {
		.format = VK_FMT,
		.samples = VK_SAMPLE_COUNT_1_BIT,
		.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
		.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
		.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
		.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
		.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
		.finalLayout = final_layout,
	};
	VkAttachmentReference ref = {
		.attachment = 0,
		.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
	};
	VkSubpassDescription sub = {
		.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
		.colorAttachmentCount = 1,
		.pColorAttachments = &ref,
	};
	VkRenderPassCreateInfo ci = {
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
		.attachmentCount = 1,
		.pAttachments = &att,
		.subpassCount = 1,
		.pSubpasses = &sub,
	};
	VkRenderPass rp = VK_NULL_HANDLE;
	vkCreateRenderPass(dev, &ci, NULL, &rp);
	return rp;
}

static bool gpu_init(struct gpu *g, VkPhysicalDevice phys,
		     const struct options *opt)
{
	memset(g, 0, sizeof(*g));
	g->phys = phys;
	g->drm_fd = -1;

	VkPhysicalDeviceProperties props;
	vkGetPhysicalDeviceProperties(phys, &props);
	snprintf(g->name, sizeof(g->name), "%s", props.deviceName);

	if (props.deviceType != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU &&
	    props.deviceType != VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) {
		fprintf(stderr, "  GPU '%s' is not a hardware GPU, skipping\n",
			g->name);
		return false;
	}

	if (!dev_supports(phys, VK_EXT_IMAGE_DRM_FORMAT_MODIFIER_EXTENSION_NAME)) {
		fprintf(stderr, "  GPU '%s' lacks DRM format modifier support, skipping\n",
			g->name);
		return false;
	}

	vkGetPhysicalDeviceMemoryProperties(phys, &g->memprops);

	/* Pick a graphics-capable queue family. */
	uint32_t nq = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(phys, &nq, NULL);
	VkQueueFamilyProperties *qf = calloc(nq, sizeof(*qf));
	vkGetPhysicalDeviceQueueFamilyProperties(phys, &nq, qf);
	bool have_q = false;
	for (uint32_t i = 0; i < nq; i++) {
		if (qf[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
			g->qfamily = i;
			g->timestamp_valid_bits = qf[i].timestampValidBits;
			have_q = true;
			break;
		}
	}
	free(qf);
	if (!have_q)
		return false;

	/* timestampPeriod is the same for the whole device. */
	g->timestamp_period = props.limits.timestampPeriod;

	/* Only enable the extensions that are actually present. */
	const char *enabled[8];
	uint32_t n_enabled = 0;
	for (size_t i = 0; i < sizeof(k_dev_exts) / sizeof(k_dev_exts[0]); i++) {
		if (dev_supports(phys, k_dev_exts[i]))
			enabled[n_enabled++] = k_dev_exts[i];
	}

	float prio = 1.0f;
	VkDeviceQueueCreateInfo qci = {
		.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
		.queueFamilyIndex = g->qfamily,
		.queueCount = 1,
		.pQueuePriorities = &prio,
	};
	VkDeviceCreateInfo dci = {
		.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
		.queueCreateInfoCount = 1,
		.pQueueCreateInfos = &qci,
		.enabledExtensionCount = n_enabled,
		.ppEnabledExtensionNames = enabled,
	};
	if (vkCreateDevice(phys, &dci, NULL, &g->dev) != VK_SUCCESS) {
		fprintf(stderr, "  vkCreateDevice failed for '%s'\n", g->name);
		return false;
	}
	vkGetDeviceQueue(g->dev, g->qfamily, 0, &g->queue);

	g->GetMemoryFdKHR =
		(PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(g->dev, "vkGetMemoryFdKHR");
	g->GetModifierProps =
		(PFN_vkGetImageDrmFormatModifierPropertiesEXT)
		vkGetDeviceProcAddr(g->dev, "vkGetImageDrmFormatModifierPropertiesEXT");

	/* DRM render node + GBM device (used for the GBM mapping path). */
	g->drm_fd = open_drm_render_node(phys);
	if (g->drm_fd >= 0)
		g->gbm = gbm_create_device(g->drm_fd);
	if (!g->gbm)
		fprintf(stderr, "  warning: no GBM device for '%s', GBM tests will be skipped\n",
			g->name);

	/* Shared pipeline resources. */
	g->vert = make_shader(g->dev, triangle_vert, sizeof(triangle_vert));
	g->frag = make_shader(g->dev, triangle_frag, sizeof(triangle_frag));

	VkPushConstantRange pcr = {
		.stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
		.offset = 0,
		.size = sizeof(float),
	};
	VkPipelineLayoutCreateInfo plci = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.pushConstantRangeCount = 1,
		.pPushConstantRanges = &pcr,
	};
	vkCreatePipelineLayout(g->dev, &plci, NULL, &g->playout);

	g->rpass_ref = make_render_pass(g->dev, VK_IMAGE_LAYOUT_GENERAL);

	VkPipelineShaderStageCreateInfo stages[2] = {
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = g->vert,
			.pName = "main",
		},
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = g->frag,
			.pName = "main",
		},
	};
	VkPipelineVertexInputStateCreateInfo vin = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
	};
	VkPipelineInputAssemblyStateCreateInfo ia = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
	};
	VkViewport vp = {
		.width = (float)opt->width,
		.height = (float)opt->height,
		.maxDepth = 1.0f,
	};
	VkRect2D scissor = { .extent = { opt->width, opt->height } };
	VkPipelineViewportStateCreateInfo vps = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.viewportCount = 1,
		.pViewports = &vp,
		.scissorCount = 1,
		.pScissors = &scissor,
	};
	VkPipelineRasterizationStateCreateInfo rs = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.polygonMode = VK_POLYGON_MODE_FILL,
		.cullMode = VK_CULL_MODE_NONE,
		.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
		.lineWidth = 1.0f,
	};
	VkPipelineMultisampleStateCreateInfo ms = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
	};
	VkPipelineColorBlendAttachmentState cba = {
		.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
				  VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
	};
	VkPipelineColorBlendStateCreateInfo cb = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.attachmentCount = 1,
		.pAttachments = &cba,
	};
	VkGraphicsPipelineCreateInfo gpci = {
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.stageCount = 2,
		.pStages = stages,
		.pVertexInputState = &vin,
		.pInputAssemblyState = &ia,
		.pViewportState = &vps,
		.pRasterizationState = &rs,
		.pMultisampleState = &ms,
		.pColorBlendState = &cb,
		.layout = g->playout,
		.renderPass = g->rpass_ref,
		.subpass = 0,
	};
	if (vkCreateGraphicsPipelines(g->dev, VK_NULL_HANDLE, 1, &gpci, NULL,
				      &g->pipeline) != VK_SUCCESS) {
		fprintf(stderr, "  pipeline creation failed for '%s'\n", g->name);
		return false;
	}

	VkCommandPoolCreateInfo cpci = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
		.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
		.queueFamilyIndex = g->qfamily,
	};
	vkCreateCommandPool(g->dev, &cpci, NULL, &g->cmdpool);

	return true;
}

static void gpu_finish(struct gpu *g)
{
	if (g->dev) {
		vkDeviceWaitIdle(g->dev);
		vkDestroyCommandPool(g->dev, g->cmdpool, NULL);
		vkDestroyPipeline(g->dev, g->pipeline, NULL);
		vkDestroyRenderPass(g->dev, g->rpass_ref, NULL);
		vkDestroyPipelineLayout(g->dev, g->playout, NULL);
		vkDestroyShaderModule(g->dev, g->vert, NULL);
		vkDestroyShaderModule(g->dev, g->frag, NULL);
		vkDestroyDevice(g->dev, NULL);
	}
	if (g->gbm)
		gbm_device_destroy(g->gbm);
	if (g->drm_fd >= 0)
		close(g->drm_fd);
}

/*
 * Modifier enumeration.
 */

static uint32_t query_modifiers(struct gpu *g, uint64_t *out, uint32_t max)
{
	VkDrmFormatModifierPropertiesListEXT list = {
		.sType = VK_STRUCTURE_TYPE_DRM_FORMAT_MODIFIER_PROPERTIES_LIST_EXT,
	};
	VkFormatProperties2 fp = {
		.sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2,
		.pNext = &list,
	};
	vkGetPhysicalDeviceFormatProperties2(g->phys, VK_FMT, &fp);
	if (list.drmFormatModifierCount == 0)
		return 0;

	VkDrmFormatModifierPropertiesEXT *props =
		calloc(list.drmFormatModifierCount, sizeof(*props));
	list.pDrmFormatModifierProperties = props;
	vkGetPhysicalDeviceFormatProperties2(g->phys, VK_FMT, &fp);

	uint32_t n = 0;
	for (uint32_t i = 0; i < list.drmFormatModifierCount && n < max; i++) {
		if (props[i].drmFormatModifierTilingFeatures &
		    VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT)
			out[n++] = props[i].drmFormatModifier;
	}
	free(props);
	return n;
}

/*
 * Determine which modifier the implementation considers optimal: create an
 * image offering the whole candidate list and let the driver pick, then read
 * back the modifier it chose.  Returns DRM_FORMAT_MOD_INVALID on failure.
 */
static uint64_t query_optimal_modifier(struct gpu *g, const struct options *opt,
				       const uint64_t *mods, uint32_t nmods)
{
	if (!g->GetModifierProps)
		return DRM_FORMAT_MOD_INVALID;

	VkImageDrmFormatModifierListCreateInfoEXT modinfo = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_LIST_CREATE_INFO_EXT,
		.drmFormatModifierCount = nmods,
		.pDrmFormatModifiers = mods,
	};
	VkImageCreateInfo ici = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
		.pNext = &modinfo,
		.imageType = VK_IMAGE_TYPE_2D,
		.format = VK_FMT,
		.extent = { opt->width, opt->height, 1 },
		.mipLevels = 1,
		.arrayLayers = 1,
		.samples = VK_SAMPLE_COUNT_1_BIT,
		.tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT,
		.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
			 VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
	};
	VkImage img;
	if (vkCreateImage(g->dev, &ici, NULL, &img) != VK_SUCCESS)
		return DRM_FORMAT_MOD_INVALID;

	VkImageDrmFormatModifierPropertiesEXT props = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_PROPERTIES_EXT,
	};
	uint64_t chosen = DRM_FORMAT_MOD_INVALID;
	if (g->GetModifierProps(g->dev, img, &props) == VK_SUCCESS)
		chosen = props.drmFormatModifier;

	vkDestroyImage(g->dev, img, NULL);
	return chosen;
}

/*
 * Per-test target setup / teardown.
 */

static bool create_image(struct gpu *g, const struct options *opt,
			 uint64_t modifier, bool exportable,
			 bool need_host_visible, struct target *t)
{
	uint64_t mods[1] = { modifier };
	VkImageDrmFormatModifierListCreateInfoEXT modinfo = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_LIST_CREATE_INFO_EXT,
		.drmFormatModifierCount = 1,
		.pDrmFormatModifiers = mods,
	};
	VkExternalMemoryImageCreateInfo extimg = {
		.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
		.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
	};
	if (exportable)
		modinfo.pNext = &extimg;

	VkImageCreateInfo ici = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
		.pNext = &modinfo,
		.imageType = VK_IMAGE_TYPE_2D,
		.format = VK_FMT,
		.extent = { opt->width, opt->height, 1 },
		.mipLevels = 1,
		.arrayLayers = 1,
		.samples = VK_SAMPLE_COUNT_1_BIT,
		.tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT,
		.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
			 VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
	};
	if (vkCreateImage(g->dev, &ici, NULL, &t->image) != VK_SUCCESS)
		return false;

	VkMemoryRequirements req;
	vkGetImageMemoryRequirements(g->dev, t->image, &req);

	VkMemoryPropertyFlags required =
		need_host_visible ? VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT : 0;
	VkMemoryPropertyFlags preferred =
		need_host_visible ? VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
				  : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
	int mt = find_mem_type(&g->memprops, req.memoryTypeBits, required, preferred);
	if (mt < 0)
		return false;	/* impossible combination (e.g. no host-visible) */

	VkMemoryDedicatedAllocateInfo ded = {
		.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO,
		.image = t->image,
	};
	VkExportMemoryAllocateInfo exp = {
		.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
		.pNext = &ded,
		.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
	};
	VkMemoryAllocateInfo mai = {
		.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		.pNext = exportable ? (void *)&exp : (void *)&ded,
		.allocationSize = req.size,
		.memoryTypeIndex = (uint32_t)mt,
	};
	if (vkAllocateMemory(g->dev, &mai, NULL, &t->mem) != VK_SUCCESS)
		return false;
	if (vkBindImageMemory(g->dev, t->image, t->mem, 0) != VK_SUCCESS)
		return false;

	t->mem_coherent = (g->memprops.memoryTypes[mt].propertyFlags &
			   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;

	VkImageSubresource sub = {
		.aspectMask = VK_IMAGE_ASPECT_MEMORY_PLANE_0_BIT_EXT,
	};
	vkGetImageSubresourceLayout(g->dev, t->image, &sub, &t->layout);
	return true;
}

static bool create_shadow(struct gpu *g, const struct options *opt, struct target *t)
{
	VkBufferCreateInfo bci = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = (VkDeviceSize)opt->width * opt->height * BYTES_PER_PIXEL,
		.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};
	if (vkCreateBuffer(g->dev, &bci, NULL, &t->shadow_buf) != VK_SUCCESS)
		return false;

	VkMemoryRequirements req;
	vkGetBufferMemoryRequirements(g->dev, t->shadow_buf, &req);

	/* The shadow buffer should be host cached if possible. */
	int mt = find_mem_type(&g->memprops, req.memoryTypeBits,
			       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
			       VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
	if (mt < 0)
		return false;

	VkMemoryAllocateInfo mai = {
		.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		.allocationSize = req.size,
		.memoryTypeIndex = (uint32_t)mt,
	};
	if (vkAllocateMemory(g->dev, &mai, NULL, &t->shadow_mem) != VK_SUCCESS)
		return false;
	vkBindBufferMemory(g->dev, t->shadow_buf, t->shadow_mem, 0);
	t->shadow_coherent = (g->memprops.memoryTypes[mt].propertyFlags &
			      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
	vkMapMemory(g->dev, t->shadow_mem, 0, VK_WHOLE_SIZE, 0, &t->shadow_map);
	return true;
}

static bool import_gbm(struct gpu *g, const struct options *opt,
		       uint64_t modifier, struct target *t)
{
	VkMemoryGetFdInfoKHR gfd = {
		.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR,
		.memory = t->mem,
		.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
	};
	if (g->GetMemoryFdKHR(g->dev, &gfd, &t->dmabuf_fd) != VK_SUCCESS)
		return false;

	struct gbm_import_fd_modifier_data data = {
		.width = opt->width,
		.height = opt->height,
		.format = DRM_FMT,
		.num_fds = 1,
		.fds[0] = t->dmabuf_fd,
		.strides[0] = (int)t->layout.rowPitch,
		.offsets[0] = (int)t->layout.offset,
		.modifier = modifier,
	};
	t->bo = gbm_bo_import(g->gbm, GBM_BO_IMPORT_FD_MODIFIER, &data,
			      GBM_BO_USE_RENDERING);
	return t->bo != NULL;
}

static bool target_create(struct gpu *g, const struct options *opt,
			  uint64_t modifier, enum mapping_method map,
			  enum shadow_mode shadow, struct target *t)
{
	memset(t, 0, sizeof(*t));
	t->dmabuf_fd = -1;

	bool exportable = (map == MAP_GBM);
	bool need_host_visible = (map == MAP_VULKAN && !shadow);

	if (!create_image(g, opt, modifier, exportable, need_host_visible, t))
		return false;

	if (map == MAP_GBM) {
		if (!import_gbm(g, opt, modifier, t))
			return false;
	} else if (!shadow) {
		if (vkMapMemory(g->dev, t->mem, 0, VK_WHOLE_SIZE, 0,
				&t->image_map) != VK_SUCCESS)
			return false;
	} else {
		if (!create_shadow(g, opt, t))
			return false;
	}

	/* The image must end up readable by the host, or staged for copy. */
	VkImageLayout final = shadow ? VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
				     : VK_IMAGE_LAYOUT_GENERAL;
	t->rpass = make_render_pass(g->dev, final);

	VkImageViewCreateInfo ivci = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
		.image = t->image,
		.viewType = VK_IMAGE_VIEW_TYPE_2D,
		.format = VK_FMT,
		.subresourceRange = {
			.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
			.levelCount = 1,
			.layerCount = 1,
		},
	};
	if (vkCreateImageView(g->dev, &ivci, NULL, &t->view) != VK_SUCCESS)
		return false;

	VkFramebufferCreateInfo fbci = {
		.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
		.renderPass = t->rpass,
		.attachmentCount = 1,
		.pAttachments = &t->view,
		.width = opt->width,
		.height = opt->height,
		.layers = 1,
	};
	if (vkCreateFramebuffer(g->dev, &fbci, NULL, &t->fb) != VK_SUCCESS)
		return false;

	VkCommandBufferAllocateInfo cbai = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		.commandPool = g->cmdpool,
		.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
		.commandBufferCount = 1,
	};
	vkAllocateCommandBuffers(g->dev, &cbai, &t->cmd);

	VkFenceCreateInfo fci = {
		.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
	};
	vkCreateFence(g->dev, &fci, NULL, &t->fence);

	if (g->timestamp_valid_bits) {
		VkQueryPoolCreateInfo qpci = {
			.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
			.queryType = VK_QUERY_TYPE_TIMESTAMP,
			.queryCount = 2,
		};
		vkCreateQueryPool(g->dev, &qpci, NULL, &t->qpool);
	}
	return true;
}

static void target_destroy(struct gpu *g, struct target *t)
{
	vkDeviceWaitIdle(g->dev);
	if (t->bo)
		gbm_bo_destroy(t->bo);
	if (t->dmabuf_fd >= 0)
		close(t->dmabuf_fd);
	if (t->shadow_map)
		vkUnmapMemory(g->dev, t->shadow_mem);
	if (t->shadow_buf)
		vkDestroyBuffer(g->dev, t->shadow_buf, NULL);
	if (t->shadow_mem)
		vkFreeMemory(g->dev, t->shadow_mem, NULL);
	if (t->image_map)
		vkUnmapMemory(g->dev, t->mem);
	if (t->qpool)
		vkDestroyQueryPool(g->dev, t->qpool, NULL);
	if (t->fence)
		vkDestroyFence(g->dev, t->fence, NULL);
	if (t->fb)
		vkDestroyFramebuffer(g->dev, t->fb, NULL);
	if (t->view)
		vkDestroyImageView(g->dev, t->view, NULL);
	if (t->rpass)
		vkDestroyRenderPass(g->dev, t->rpass, NULL);
	if (t->image)
		vkDestroyImage(g->dev, t->image, NULL);
	if (t->mem)
		vkFreeMemory(g->dev, t->mem, NULL);
}

/*
 * Rendering and capture.
 */

static void record_frame(struct gpu *g, const struct options *opt,
			 struct target *t, enum shadow_mode shadow, float angle)
{
	vkResetCommandBuffer(t->cmd, 0);

	VkCommandBufferBeginInfo bi = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};
	vkBeginCommandBuffer(t->cmd, &bi);

	if (t->qpool) {
		vkCmdResetQueryPool(t->cmd, t->qpool, 0, 2);
		vkCmdWriteTimestamp(t->cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
				    t->qpool, 0);
	}

	VkClearValue clear = { .color = { .float32 = { 0.05f, 0.05f, 0.08f, 1.0f } } };
	VkRenderPassBeginInfo rbi = {
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
		.renderPass = t->rpass,
		.framebuffer = t->fb,
		.renderArea = { .extent = { opt->width, opt->height } },
		.clearValueCount = 1,
		.pClearValues = &clear,
	};
	vkCmdBeginRenderPass(t->cmd, &rbi, VK_SUBPASS_CONTENTS_INLINE);
	vkCmdBindPipeline(t->cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, g->pipeline);
	vkCmdPushConstants(t->cmd, g->playout, VK_SHADER_STAGE_VERTEX_BIT, 0,
			   sizeof(float), &angle);
	vkCmdDraw(t->cmd, 3, 1, 0, 0);
	vkCmdEndRenderPass(t->cmd);

	if (shadow) {
		VkBufferImageCopy region = {
			.imageSubresource = {
				.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
				.layerCount = 1,
			},
			.imageExtent = { opt->width, opt->height, 1 },
		};
		vkCmdCopyImageToBuffer(t->cmd, t->image,
				       VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
				       t->shadow_buf, 1, &region);

		VkMemoryBarrier mb = {
			.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
			.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
			.dstAccessMask = VK_ACCESS_HOST_READ_BIT,
		};
		vkCmdPipelineBarrier(t->cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
				     VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0,
				     NULL, 0, NULL);
	} else {
		/* Make the colour writes visible to host reads of the map. */
		VkMemoryBarrier mb = {
			.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
			.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
			.dstAccessMask = VK_ACCESS_HOST_READ_BIT,
		};
		vkCmdPipelineBarrier(t->cmd,
				     VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
				     VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0,
				     NULL, 0, NULL);
	}

	if (t->qpool)
		vkCmdWriteTimestamp(t->cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
				    t->qpool, 1);

	vkEndCommandBuffer(t->cmd);
}

/* Copy the rendered frame into the host buffer.  Returns false on error. */
static bool capture_frame(struct gpu *g, const struct options *opt,
			  struct target *t, enum mapping_method map,
			  enum shadow_mode shadow, uint8_t *dst)
{
	const uint32_t w = opt->width, h = opt->height;
	const size_t dst_pitch = (size_t)w * BYTES_PER_PIXEL;
	const uint8_t *src;
	size_t src_pitch;

	if (map == MAP_GBM) {
		void *bo_map = NULL;
		uint32_t stride = 0;
		void *data = gbm_bo_map(t->bo, 0, 0, w, h, GBM_BO_TRANSFER_READ,
					&stride, &bo_map);
		if (data == MAP_FAILED || data == NULL)
			return false;
		src = data;
		src_pitch = stride;
		for (uint32_t y = 0; y < h; y++)
			memcpy(dst + y * dst_pitch, src + (size_t)y * src_pitch,
			       dst_pitch);
		gbm_bo_unmap(t->bo, bo_map);
	} else if (!shadow) {
		if (!t->mem_coherent) {
			VkMappedMemoryRange r = {
				.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
				.memory = t->mem,
				.offset = 0,
				.size = VK_WHOLE_SIZE,
			};
			vkInvalidateMappedMemoryRanges(g->dev, 1, &r);
		}
		src = (const uint8_t *)t->image_map + t->layout.offset;
		src_pitch = t->layout.rowPitch;
		for (uint32_t y = 0; y < h; y++)
			memcpy(dst + y * dst_pitch, src + (size_t)y * src_pitch,
			       dst_pitch);
	} else {
		if (!t->shadow_coherent) {
			VkMappedMemoryRange r = {
				.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
				.memory = t->shadow_mem,
				.offset = 0,
				.size = VK_WHOLE_SIZE,
			};
			vkInvalidateMappedMemoryRanges(g->dev, 1, &r);
		}
		/* The copy region is tightly packed. */
		memcpy(dst, t->shadow_map, dst_pitch * h);
	}

	/* Ensure the copy cannot be optimised away. */
	g_sink ^= ((const uint64_t *)dst)[0];
	return true;
}

static const char *map_name(enum mapping_method m)
{
	return m == MAP_GBM ? "gbm" : "vulkan";
}

static struct json_object *result_skip(struct json_object *o,
				       enum mapping_method map,
				       enum shadow_mode shadow,
				       const char *modname, const char *reason)
{
	json_object_object_add(o, "skipped", json_object_new_boolean(true));
	json_object_object_add(o, "skip_reason", json_object_new_string(reason));
	fprintf(stderr, "  [skip] %s / %s / shadow=%d : %s\n", modname,
		map_name(map), shadow, reason);
	return o;
}

/* Runs one test combination and returns the populated JSON result object. */
static struct json_object *run_test(const struct test_context *ctx,
				    enum mapping_method map,
				    enum shadow_mode shadow)
{
	struct gpu *g = ctx->gpu;
	const struct options *opt = ctx->opt;
	uint64_t modifier = ctx->modifier;
	const char *modname = ctx->modname;
	struct best *best = ctx->best;

	struct json_object *o = json_object_new_object();
	json_object_object_add(o, "modifier", json_object_new_string(modname));
	char hex[32];
	snprintf(hex, sizeof(hex), "0x%016" PRIx64, modifier);
	json_object_object_add(o, "modifier_value", json_object_new_string(hex));
	json_object_object_add(o, "optimal_tiling",
			       json_object_new_boolean(ctx->optimal));
	json_object_object_add(o, "mapping",
			       json_object_new_string(map_name(map)));
	json_object_object_add(o, "shadow_buffer", json_object_new_boolean(shadow));

	/* Impossible combinations. */
	if (map == MAP_GBM && shadow)
		return result_skip(o, map, shadow, modname,
				   "gbm mapping never uses a shadow buffer");
	if (map == MAP_GBM && !g->gbm)
		return result_skip(o, map, shadow, modname,
				   "no GBM device available");
	/* Directly mapping the image (no shadow buffer, no gbm detiling) only
	 * yields correct pixels for a linear layout; tiled modifiers would be
	 * read back as garbage. */
	if (map == MAP_VULKAN && !shadow && modifier != DRM_FORMAT_MOD_LINEAR)
		return result_skip(o, map, shadow, modname,
				   "vulkan mapping without a shadow buffer only works for the linear modifier");

	fprintf(stderr, "  [run ] %s / %s / shadow=%d ... ", modname,
		map_name(map), shadow);
	fflush(stderr);

	struct target t;
	if (!target_create(g, opt, modifier, map, shadow, &t)) {
		target_destroy(g, &t);
		const char *reason;
		if (map == MAP_GBM)
			reason = "modifier not importable via gbm (e.g. compressed/DCC)";
		else if (map == MAP_VULKAN && !shadow)
			reason = "image memory is not host visible";
		else
			reason = "resource setup failed";
		fprintf(stderr, "skipped (%s)\n", reason);
		json_object_object_add(o, "skipped", json_object_new_boolean(true));
		json_object_object_add(o, "skip_reason",
				       json_object_new_string(reason));
		return o;
	}

	uint8_t *host = malloc((size_t)opt->width * opt->height * BYTES_PER_PIXEL);
	if (!host) {
		target_destroy(g, &t);
		fprintf(stderr, "skipped (oom)\n");
		json_object_object_add(o, "skipped", json_object_new_boolean(true));
		json_object_object_add(o, "skip_reason",
				       json_object_new_string("out of memory"));
		return o;
	}

	/* MJPEG output (verification only). */
	char output[256] = { 0 };
	FILE *out = NULL;
	tjhandle tj = NULL;
	if (opt->encode) {
		/* Sanitise only the modifier name (it may contain commas,
		 * spaces and slashes); the user prefix may be a real path. */
		char safe_mod[128];
		snprintf(safe_mod, sizeof(safe_mod), "%s", modname);
		for (char *p = safe_mod; *p; p++)
			if (*p == '/' || *p == ' ' || *p == ',')
				*p = '_';
		snprintf(output, sizeof(output), "%s_%s_%s_%s.mjpeg",
			 opt->out_prefix, safe_mod, map_name(map),
			 shadow ? "shadow" : "noshadow");
		out = fopen(output, "wb");
		tj = tjInitCompress();
	}

	/* Mask for the valid low bits of a timestamp, and microseconds per tick. */
	uint64_t ts_mask = t.qpool && g->timestamp_valid_bits < 64
				   ? ((uint64_t)1 << g->timestamp_valid_bits) - 1
				   : UINT64_MAX;
	double us_per_tick = g->timestamp_period / 1000.0;
	double gpu_us_total = 0;

	const float angle_step = (float)(2.0 * M_PI / 60.0);
	double start = now_sec();
	double cpu_start = now_cpu_sec();
	uint64_t frame = 0;

	while (now_sec() - start < opt->duration) {
		record_frame(g, opt, &t, shadow, frame * angle_step);

		vkResetFences(g->dev, 1, &t.fence);
		VkSubmitInfo si = {
			.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
			.commandBufferCount = 1,
			.pCommandBuffers = &t.cmd,
		};
		vkQueueSubmit(g->queue, 1, &si, t.fence);
		vkWaitForFences(g->dev, 1, &t.fence, VK_TRUE, UINT64_MAX);

		if (t.qpool) {
			uint64_t ts[2];
			if (vkGetQueryPoolResults(g->dev, t.qpool, 0, 2,
						  sizeof(ts), ts, sizeof(ts[0]),
						  VK_QUERY_RESULT_64_BIT) == VK_SUCCESS)
				gpu_us_total += ((ts[1] & ts_mask) -
						 (ts[0] & ts_mask)) * us_per_tick;
		}

		if (!capture_frame(g, opt, &t, map, shadow, host)) {
			fprintf(stderr, "capture failed ... ");
			break;
		}

		if (tj) {
			unsigned char *jpeg = NULL;
			unsigned long jpeg_size = 0;
			if (tjCompress2(tj, host, opt->width, 0, opt->height,
					TJPF_BGRX, &jpeg, &jpeg_size, TJSAMP_420,
					85, 0) == 0) {
				fwrite(jpeg, 1, jpeg_size, out);
				tjFree(jpeg);
			}
		}
		frame++;
	}

	double cpu_total = now_cpu_sec() - cpu_start;
	double elapsed = now_sec() - start;
	double fps = elapsed > 0 ? frame / elapsed : 0;
	/* CPU time consumed per frame, and CPU usage as a share of wall time
	 * (it can exceed 100% when the Vulkan driver uses worker threads). */
	double cpu_us_per_frame = frame ? cpu_total * 1e6 / frame : 0;
	double cpu_usage_percent = elapsed > 0 ? cpu_total / elapsed * 100 : 0;
	/* GPU time spent in our Vulkan command buffer (the render, plus the
	 * image->buffer copy on the shadow path), and the GPU usage it
	 * represents as a share of wall time.  This does not cover work done
	 * inside gbm_bo_map, which may detile on either the CPU or GPU and is
	 * invisible to these queries. */
	double vk_gpu_us_per_frame = frame ? gpu_us_total / frame : 0;
	double vk_gpu_usage_percent =
		elapsed > 0 ? gpu_us_total / (elapsed * 1e6) * 100 : 0;
	if (t.qpool)
		fprintf(stderr, "%" PRIu64 " frames, %.1f fps, %.1f us/frame cpu (%.0f%% cpu usage), %.1f us/frame vk-gpu (%.0f%% vk-gpu usage)\n",
			frame, fps, cpu_us_per_frame, cpu_usage_percent,
			vk_gpu_us_per_frame, vk_gpu_usage_percent);
	else
		fprintf(stderr, "%" PRIu64 " frames, %.1f fps, %.1f us/frame cpu (%.0f%% cpu usage)\n",
			frame, fps, cpu_us_per_frame, cpu_usage_percent);

	if (!best->valid || fps > best->fps) {
		best->valid = true;
		best->fps = fps;
		snprintf(best->gpu, sizeof(best->gpu), "%s", g->name);
		snprintf(best->modifier, sizeof(best->modifier), "%s", modname);
		best->map = map;
		best->shadow = shadow;
	}

	bool measured_gpu = t.qpool != NULL;

	if (tj)
		tjDestroy(tj);
	if (out)
		fclose(out);
	free(host);
	target_destroy(g, &t);

	json_object_object_add(o, "skipped", json_object_new_boolean(false));
	json_object_object_add(o, "frames", json_object_new_int64((int64_t)frame));
	json_object_object_add(o, "fps", json_object_new_double(fps));
	json_object_object_add(o, "cpu_us_per_frame",
			       json_object_new_double(cpu_us_per_frame));
	json_object_object_add(o, "cpu_usage_percent",
			       json_object_new_double(cpu_usage_percent));
	if (measured_gpu) {
		json_object_object_add(o, "vk_gpu_us_per_frame",
				       json_object_new_double(vk_gpu_us_per_frame));
		json_object_object_add(o, "vk_gpu_usage_percent",
				       json_object_new_double(vk_gpu_usage_percent));
	}
	if (output[0])
		json_object_object_add(o, "output", json_object_new_string(output));
	return o;
}

/* Read a single field's value from a "KEY=VALUE" or "key : value" file. */
static void read_field(const char *path, const char *key, char prefix_char,
		       char *out, size_t len)
{
	out[0] = '\0';
	FILE *f = fopen(path, "r");
	if (!f)
		return;

	size_t keylen = strlen(key);
	char line[512];
	while (fgets(line, sizeof(line), f)) {
		if (strncmp(line, key, keylen) != 0)
			continue;
		char *v = strchr(line + keylen, prefix_char);
		if (!v)
			break;
		v++;
		while (*v == ' ' || *v == '\t' || *v == '"')
			v++;
		v[strcspn(v, "\n")] = '\0';
		char *end = v + strlen(v);
		while (end > v && (end[-1] == '"' || end[-1] == ' '))
			*--end = '\0';
		snprintf(out, len, "%s", v);
		break;
	}
	fclose(f);
}

static struct json_object *system_info(void)
{
	struct json_object *o = json_object_new_object();
	char buf[256];

	read_field("/etc/os-release", "PRETTY_NAME", '=', buf, sizeof(buf));
	json_object_object_add(o, "distro", json_object_new_string(buf));

	struct utsname u;
	if (uname(&u) == 0) {
		snprintf(buf, sizeof(buf), "%s %s", u.sysname, u.release);
		json_object_object_add(o, "kernel", json_object_new_string(buf));
	}

	read_field("/proc/cpuinfo", "model name", ':', buf, sizeof(buf));
	json_object_object_add(o, "cpu", json_object_new_string(buf));
	json_object_object_add(o, "cpu_threads",
			       json_object_new_int(sysconf(_SC_NPROCESSORS_ONLN)));

	int64_t mem = (int64_t)sysconf(_SC_PHYS_PAGES) * sysconf(_SC_PAGE_SIZE);
	json_object_object_add(o, "memory_bytes", json_object_new_int64(mem));

	return o;
}

#define STR(x) #x
#define XSTR(x) STR(x)

#define MEMORY_PROPERTY_FLAGS(X) \
	X(DEVICE_LOCAL_BIT) \
	X(HOST_VISIBLE_BIT) \
	X(HOST_COHERENT_BIT) \
	X(HOST_CACHED_BIT) \
	X(LAZILY_ALLOCATED_BIT) \
	X(PROTECTED_BIT) \
	X(DEVICE_COHERENT_BIT_AMD) \
	X(DEVICE_UNCACHED_BIT_AMD) \
	X(RDMA_CAPABLE_BIT_NV)

static struct json_object *memory_types_json(const VkPhysicalDeviceMemoryProperties *mp)
{
	struct json_object *arr = json_object_new_array();
	for (uint32_t i = 0; i < mp->memoryTypeCount; i++) {
		VkMemoryPropertyFlags pf = mp->memoryTypes[i].propertyFlags;

		struct json_object *o = json_object_new_object();
		json_object_object_add(o, "index", json_object_new_int((int)i));
		json_object_object_add(o, "heap",
			json_object_new_int((int)mp->memoryTypes[i].heapIndex));

		struct json_object *props = json_object_new_array();
#define X(name) \
		if (pf & VK_MEMORY_PROPERTY_##name) \
			json_object_array_add(props, \
				json_object_new_string(XSTR(name)));
		MEMORY_PROPERTY_FLAGS(X)
#undef X
		json_object_object_add(o, "properties", props);
		json_object_array_add(arr, o);
	}
	return arr;
}

#undef MEMORY_PROPERTY_FLAGS

static void usage(const char *argv0)
{
	fprintf(stderr,
		"Usage: %s [options]\n"
		"  -d, --duration SEC   test duration per combination (default 1.0)\n"
		"  -W, --width PX        render width (default 1920)\n"
		"  -H, --height PX       render height (default 1080)\n"
		"  -e, --encode          write MJPEG output for verification\n"
		"  -o, --output PREFIX   output filename prefix (default 'capture')\n"
		"  -h, --help            show this help\n",
		argv0);
}

int main(int argc, char **argv)
{
	struct options opt = {
		.duration = 1.0,
		.width = 1920,
		.height = 1080,
		.encode = false,
		.out_prefix = "capture",
	};

	static const struct option longopts[] = {
		{ "duration", required_argument, NULL, 'd' },
		{ "width", required_argument, NULL, 'W' },
		{ "height", required_argument, NULL, 'H' },
		{ "encode", no_argument, NULL, 'e' },
		{ "output", required_argument, NULL, 'o' },
		{ "help", no_argument, NULL, 'h' },
		{ 0, 0, 0, 0 },
	};
	int c;
	while ((c = getopt_long(argc, argv, "d:W:H:eo:h", longopts, NULL)) != -1) {
		switch (c) {
		case 'd': opt.duration = atof(optarg); break;
		case 'W': opt.width = (uint32_t)atoi(optarg); break;
		case 'H': opt.height = (uint32_t)atoi(optarg); break;
		case 'e': opt.encode = true; break;
		case 'o': opt.out_prefix = optarg; break;
		case 'h': usage(argv[0]); return 0;
		default: usage(argv[0]); return 1;
		}
	}

	VkApplicationInfo ai = {
		.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
		.pApplicationName = "screen-capture-bench",
		.apiVersion = VK_API_VERSION_1_2,
	};
	VkInstanceCreateInfo ici = {
		.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
		.pApplicationInfo = &ai,
	};
	VkInstance inst;
	if (vkCreateInstance(&ici, NULL, &inst) != VK_SUCCESS) {
		fprintf(stderr, "failed to create Vulkan instance\n");
		return 1;
	}

	uint32_t ndev = 0;
	vkEnumeratePhysicalDevices(inst, &ndev, NULL);
	if (ndev == 0) {
		fprintf(stderr, "no Vulkan devices found\n");
		return 1;
	}
	VkPhysicalDevice *devs = calloc(ndev, sizeof(*devs));
	vkEnumeratePhysicalDevices(inst, &ndev, devs);

	/* The result JSON is assembled incrementally as the tests run. */
	struct json_object *root = json_object_new_object();
	json_object_object_add(root, "system", system_info());
	struct json_object *gpus = json_object_new_array();
	json_object_object_add(root, "gpus", gpus);
	struct best *bests = calloc(ndev, sizeof(*bests));

	for (uint32_t i = 0; i < ndev; i++) {
		struct gpu g;
		VkPhysicalDeviceProperties props;
		vkGetPhysicalDeviceProperties(devs[i], &props);
		fprintf(stderr, "GPU %u: %s\n", i, props.deviceName);

		if (!gpu_init(&g, devs[i], &opt)) {
			gpu_finish(&g);
			continue;
		}

		struct json_object *gpu_obj = json_object_new_object();
		json_object_object_add(gpu_obj, "gpu",
				       json_object_new_string(g.name));
		json_object_object_add(gpu_obj, "memory_types",
				       memory_types_json(&g.memprops));
		struct json_object *results = json_object_new_array();
		json_object_object_add(gpu_obj, "results", results);
		json_object_array_add(gpus, gpu_obj);

		uint64_t mods[64];
		uint32_t nmods = query_modifiers(&g, mods, 64);
		if (nmods == 0)
			fprintf(stderr, "  no usable modifiers for the format\n");

		uint64_t optimal_mod = query_optimal_modifier(&g, &opt, mods, nmods);
		if (optimal_mod != DRM_FORMAT_MOD_INVALID) {
			char modname[128];
			fprintf(stderr, "  optimal tiling modifier: %s\n",
				modifier_name(optimal_mod, modname, sizeof(modname)));
		}

		for (uint32_t m = 0; m < nmods; m++) {
			char modname[128];
			modifier_name(mods[m], modname, sizeof(modname));

			struct test_context ctx = {
				.gpu = &g,
				.opt = &opt,
				.modifier = mods[m],
				.modname = modname,
				.optimal = (mods[m] == optimal_mod),
				.best = &bests[i],
			};

			json_object_array_add(results,
				run_test(&ctx, MAP_GBM, SHADOW_NONE));
			json_object_array_add(results,
				run_test(&ctx, MAP_GBM, SHADOW_USED));
			json_object_array_add(results,
				run_test(&ctx, MAP_VULKAN, SHADOW_NONE));
			json_object_array_add(results,
				run_test(&ctx, MAP_VULKAN, SHADOW_USED));
		}

		gpu_finish(&g);
	}

	for (uint32_t i = 0; i < ndev; i++) {
		const struct best *b = &bests[i];
		if (!b->valid)
			continue;
		fprintf(stderr, "\nBest combination for %s:\n", b->gpu);
		fprintf(stderr, " modifier: %s\n", b->modifier);
		fprintf(stderr, " mapping: %s\n",
			b->map == MAP_GBM ? "GBM" : "Vulkan");
		fprintf(stderr, " shadow: %s\n",
			b->shadow == SHADOW_USED ? "used" : "none");
		fprintf(stderr, " fps: %.1f\n", b->fps);
	}
	free(bests);

	if (opt.encode) {
		/* A bare stream of concatenated JPEGs auto-probes as a single
		 * still image, so players must be told it is a 60 fps MJPEG
		 * video, e.g.:
		 *   mpv --demuxer-lavf-format=mjpeg --demuxer-lavf-o=framerate=60 FILE
		 *   ffplay -f mjpeg -framerate 60 FILE
		 */
		fprintf(stderr,
			"\nMJPEG written. Play e.g. with:\n"
			"  mpv --demuxer-lavf-format=mjpeg --demuxer-lavf-o=framerate=60 FILE\n"
			"  ffplay -f mjpeg -framerate 60 FILE\n");
	}

	printf("%s\n", json_object_to_json_string_ext(root, JSON_C_TO_STRING_PRETTY));
	json_object_put(root);

	free(devs);
	vkDestroyInstance(inst, NULL);
	return 0;
}
