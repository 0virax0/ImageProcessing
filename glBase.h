#pragma once
//header guard
#ifndef GLBASE_H
#define GLBASE_H


	class glWrapper : RAIIscope{
			SDL_Window *window;
			SDL_GLContext SDLcontext;
			GLuint programID;
			GLuint vertexbuffer;
			GLuint UVbuffer;
	public: glWrapper();
			int initContext(cl_platform_id platform, cl_device_id* device);
			int initGl();

			cl_mem createGLtexture(size_t width, size_t height, cl_mem_flags flags, GLuint* outTextID);

			int releaseTexture(cl_mem imageCL);
			int acquireTexture(cl_mem imageCL);

			int glOps(image* img);
			int Cleanup();
	};

#endif