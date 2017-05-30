#define ONLY_GL_BASE_HEADERS
#include "clBase.h"
#include "glBase.h"

#pragma region class glWrapper
	glWrapper::glWrapper() : RAIIscope() {}

	 int glWrapper::initContext(cl_platform_id platform, cl_device_id* device) {
		//sdl video subsystem
		if (SDL_Init(SDL_INIT_VIDEO) < 0)
		{
			std::cout << "Failed to init SDL\n";
			return false;
		}
		//create Window
		window = SDL_CreateWindow(
			"gpuMonitor",
			SDL_WINDOWPOS_CENTERED,
			SDL_WINDOWPOS_CENTERED,
			1920,
			1080,
			SDL_WINDOW_OPENGL
		);
		if (!window)
		{
			std::cout << "Unable to create window\n" << SDL_GetError();
			Cleanup();
			return FAILURE;
		}
		//Set sdl attributes
		SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
		SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
		//create context
		SDLcontext = SDL_GL_CreateContext(window);
		//glew
		glewExperimental = GL_TRUE;
		glewInit(); 

		// Create CL context properties, add WGL context & handle to DC
		cl_context_properties properties[] = {
			CL_GL_CONTEXT_KHR, (cl_context_properties)wglGetCurrentContext(), // WGL Context
			CL_WGL_HDC_KHR, (cl_context_properties)wglGetCurrentDC(), // WGL HDC
			CL_CONTEXT_PLATFORM, (cl_context_properties)platform, // OpenCL platform
			0
		};
		//pass context to cl
		cl_int err;
		context = clCreateContext(properties, 1, device, NULL, NULL, &err);
		printErr(err, "context creation failed");

		//opengl drawing initialization
		initGl();

		return SUCCESS;
	}

	 int glWrapper::initGl() {
		programID = LoadShaders("C:\\Users\\stud3aii_2\\Desktop\\HelloWorld\\shaders\\vertex.glsl", "C:\\Users\\stud3aii_2\\Desktop\\HelloWorld\\shaders\\fragment.glsl");

		//define VAO
		GLuint VAO;
		glGenVertexArrays(1, &VAO);
		glBindVertexArray(VAO);
		//define VBO
		static const GLfloat g_vertex_buffer_data[] = {
			-1.0f, -1.0f, 0.0f,
			-1.0f, 1.0f, 0.0f,		
			1.0f, 1.0f, 0.0f,
			1.0f, 1.0f, 0.0f,
			1.0f, -1.0f, 0.0f,
			-1.0f, -1.0f, 0.0f,
		};
		// put into vertex buffer
		glGenBuffers(1, &vertexbuffer);
		new destroyable<GLuint>(&vertexbuffer, this);
		glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
		glBufferData(GL_ARRAY_BUFFER, sizeof(g_vertex_buffer_data), g_vertex_buffer_data, GL_STATIC_DRAW);

		static const GLfloat g_UV_buffer_data[] = {
			-1.0f, -1.0f,
			-1.0f, 0.1f, 
			0.1f, 0.1f,
			0.1f, 0.1f,
			0.1f, -1.0f, 
			-1.0f, -1.0f, 	
		};
		// put into vertex buffer
		glGenBuffers(1, &UVbuffer);
		new destroyable<GLuint>(&UVbuffer, this);
		glBindBuffer(GL_ARRAY_BUFFER, UVbuffer);
		glBufferData(GL_ARRAY_BUFFER, sizeof(g_UV_buffer_data), g_UV_buffer_data, GL_STATIC_DRAW);

		if (glGetError() != GL_NO_ERROR)std::cout << glGetError() << std::endl;
		return SUCCESS;
	}

	 cl_mem glWrapper::createGLtexture(size_t width, size_t height, cl_mem_flags flags, GLuint* outTextID) {
		// Create one OpenGL texture
		GLuint textureID;
		glGenTextures(1, &textureID);

		// Bind texture
		glBindTexture(GL_TEXTURE_2D, textureID);

		// Give the empty image to OpenGL
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_INT_8_8_8_8, 0);

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

		//to cl image
		cl_mem imageCL = clCreateFromGLTexture2D(context, flags,
			GL_TEXTURE_2D, 0, textureID, &err);
		printErr(err, "failed to convert texture to cl image");

		//make sure the image can be used by cl kernel
		releaseTexture(imageCL);

		*outTextID = textureID;
		if (glGetError() != GL_NO_ERROR)std::cout << glGetError() << std::endl;
		return imageCL;
	}

	 int glWrapper::releaseTexture(cl_mem imageCL) {
		glFinish();
		return printErr(clEnqueueAcquireGLObjects(commandQueue, 1, &imageCL, 0, NULL, NULL), "set owning to cl failed");
	}
	 int glWrapper::acquireTexture(cl_mem imageCL) {
		int err = clEnqueueReleaseGLObjects(commandQueue, 1, &imageCL, 0, 0, 0);
		clFinish(commandQueue); //emptyes the host cl queue and wait for end, ready for the gl's;
		if (glGetError() != GL_NO_ERROR)std::cout << glGetError() << std::endl;
		return printErr(err, "acquireTexture failed");
	}

	 int glWrapper::glOps(image* img) {
		//prints the image
		glUseProgram(programID);
		glClearColor(0.0, 0.0, 1.0, 1.0);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


		//DRAW QUAD
		//set Image texture unit
		acquireTexture(img->outputImage);
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, img->outTextureID);

		// 1rst attribute buffer : vertices
		glEnableVertexAttribArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

		// 2nd attribute buffer : UV
		glEnableVertexAttribArray(1);
		glBindBuffer(GL_ARRAY_BUFFER, UVbuffer);
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, (void*)0);

		glDrawArrays(GL_TRIANGLES, 0, 6);
		glDisableVertexAttribArray(0);
		glDisableVertexAttribArray(1);

		SDL_GL_SwapWindow(window);
		releaseTexture(img->outputImage);

		glFinish();
		return SUCCESS;
	} 
	 int glWrapper::Cleanup()
	{
		SDL_GL_DeleteContext(SDLcontext);
		SDL_DestroyWindow(window);
		SDL_Quit();
		glDeleteProgram(programID);
		std::cout << "free gl" << std::endl;
		return 1;
	}
#pragma endregion
