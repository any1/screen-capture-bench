project(
  'screen-capture-bench',
  'c',
  version: '0.1.0',
  default_options: ['c_std=c11', 'warning_level=2'],
)

cc = meson.get_compiler('c')

gbm_dep = dependency('gbm')
drm_dep = dependency('libdrm')
vulkan_dep = dependency('vulkan')
jsonc_dep = dependency('json-c')
turbojpeg_dep = dependency('libturbojpeg')
m_dep = cc.find_library('m', required: false)

glslang = find_program('glslangValidator')

shader_headers = []
foreach s : [
  ['src/shaders/triangle.vert', 'triangle_vert'],
  ['src/shaders/triangle.frag', 'triangle_frag'],
]
  shader_headers += custom_target(
    s[1],
    input: s[0],
    output: s[1] + '.h',
    command: [glslang, '-V', '--vn', s[1], '-o', '@OUTPUT@', '@INPUT@'],
  )
endforeach

executable(
  'screen-capture-bench',
  ['src/main.c', shader_headers],
  dependencies: [gbm_dep, drm_dep, vulkan_dep, jsonc_dep, turbojpeg_dep, m_dep],
  install: true,
)
