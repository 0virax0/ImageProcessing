#version 410 core
#extension GL_ARB_explicit_uniform_location : require
layout(location = 0) in vec4 coord4d;
layout(location = 1) uniform mat4 mvp_matrix;
out vec2 UV;
void main(){
	gl_PointSize = 1.0;
	UV = vec2(((gl_VertexID % 2048) / 2048.0f * (4096.0f/6656.0f)+1280.0f/6656.0f), -(floor(gl_VertexID / 2048) / 1024.0f * (2048.0f/3328.0f)+640.0f/3328.0f));
	gl_Position = mvp_matrix * coord4d;
}