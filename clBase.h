#pragma once
//header guard
#ifndef CLBASE_H
#define CLBASE_H

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <iostream>
#include <fstream>
#include <turbojpeg.h>
//opengl
#include <CL\cl_gl_ext.h>
#include <gl\glew.h>
#include <SDL.h>
#include <SDL_opengl.h>
#include <gl\glu.h>
#include "SDKUtil.hpp"

#define SUCCESS 1
#define FAILURE 0
#define cl_khr_gl_sharing 1

template<typename T>
const char* checkCodeStr(T input);

template<typename T>
int printErr(T input, std::string specs);

int convertToString(const char *filename, std::string* s);
GLuint LoadShaders(const char * vertex_file_path, const char * fragment_file_path);

//global variables
	extern cl_context context;
	extern cl_int err;
	extern cl_command_queue commandQueue;

	extern class glWrapper;
//resource handling
	class innerDestroyable {
	public:
		virtual void destroy()=0;
	};
	class destroyer {
	public:
		std::vector<innerDestroyable*> arr;
		destroyer();
		~destroyer();
	};
	class RAIIscope {
	public:
		destroyer* localRAII;
		RAIIscope();
		~RAIIscope();
	};
	template <typename T> 
	class destroyable : innerDestroyable {
	public:
		T* reference;
		int localRAIIIndex;
		destroyable(T* p, RAIIscope* scope);
		void destroy() override;

		int selfDestroy(T * obj);
		~destroyable(); 		//if it destroys before time, take the pointer
	};

//image class
	class image : RAIIscope {
	public: 
			size_t size;
			size_t height;
			size_t width;
			int jpegSubsamp;
			cl_mem inputImage;
			cl_mem outputImage;
			GLuint outTextureID;
			unsigned char* imageBuffer;
			unsigned char* imageOutBuffer;
			cl_image_format format;
			static const int COLOR_COMPONENTS = 4;

			image();
			int loadImage(const char *filename);
			int fetchOutImage();

	};
#endif