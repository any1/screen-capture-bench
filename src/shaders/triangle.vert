#version 450

layout(push_constant) uniform Push {
	float angle;
} pc;

layout(location = 0) out vec3 v_color;

vec2 positions[3] = vec2[](
	vec2( 0.0, -0.6),
	vec2( 0.6,  0.5),
	vec2(-0.6,  0.5)
);

vec3 colors[3] = vec3[](
	vec3(1.0, 0.0, 0.0),
	vec3(0.0, 1.0, 0.0),
	vec3(0.0, 0.0, 1.0)
);

void main()
{
	float c = cos(pc.angle);
	float s = sin(pc.angle);
	mat2 rot = mat2(c, -s, s, c);
	gl_Position = vec4(rot * positions[gl_VertexIndex], 0.0, 1.0);
	v_color = colors[gl_VertexIndex];
}
