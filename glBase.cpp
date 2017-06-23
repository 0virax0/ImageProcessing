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
			screenWidth,
			screenHeight,
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
	    char sh1[100]; strcpy(sh1, THIS_FOLDER); strcat(sh1, "\\shaders\\vertexMonitor.glsl");
		char sh2[100]; strcpy(sh2, THIS_FOLDER); strcat(sh2, "\\shaders\\fragmentMonitor.glsl");
		programID = LoadShaders(sh1, sh2);

		strcpy(sh1, THIS_FOLDER); strcat(sh1, "\\shaders\\vertexRender.glsl");
		strcpy(sh2, THIS_FOLDER); strcat(sh2, "\\shaders\\fragmentRender.glsl");
		programRenderID = LoadShaders(sh1, sh2);



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
			-1.0f, 0.0f,
			-1.0f, -1.0f, 
			0.0f, -1.0f,
			0.0f, -1.0f,
			0.0f, 0.0f, 
			-1.0f, 0.0f, 	
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
		return printErr(err, "acquireTexture/buffer failed");
	}

	 cl_mem glWrapper::createGLbuffer(size_t width, size_t height, cl_mem_flags flags) {
		 // Create one OpenGL texture
		 //define VAO
		 GLuint VAO;
		 glGenVertexArrays(1, &VAO);
		 glBindVertexArray(VAO);

		 GLuint bufferID;

		 glGenBuffers(1, &bufferID);
		 new destroyable<GLuint>(&bufferID, this);
		 glBindBuffer(GL_ARRAY_BUFFER, bufferID);
		 glBufferData(GL_ARRAY_BUFFER, sizeof(float)*4 * width * height, 0, GL_DYNAMIC_DRAW);

		 //to cl buffer
		 cl_mem bufferCL = clCreateFromGLBuffer(context, flags,
			 bufferID, &err);
		 printErr(err, "failed to convert buffer to cl buffer");

		 //make sure the buffer can be used by cl kernel
		 releaseTexture(bufferCL);		//same for buffers

		 vertexBuffer3d = bufferID;
		 if (glGetError() != GL_NO_ERROR)std::cout << glGetError() << std::endl;
		 return bufferCL;
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

	 int glWrapper::glRender(cl_mem buffer) {
		 glEnable(GL_PROGRAM_POINT_SIZE_ARB);
		 acquireTexture(buffer);
		 
		 int lastTime = 0;
		 while (true) {
			 loopRender();

			 int newTime = SDL_GetTicks();
			 char buffer[33]; itoa((int)1000/(newTime- lastTime), buffer, 10); 
			 char str[100]; strcpy(str, "GPU monitor : "); strcat(str, buffer);
			 SDL_SetWindowTitle(window, str);
			 lastTime = newTime;
		 }		
		 if (glGetError() != GL_NO_ERROR)std::cout << glGetError() << std::endl;

		 glFinish();

		 return SUCCESS;
	 }
	 void glWrapper::loopRender() {
		 //matrix
		 float angle = SDL_GetTicks() / 1000.0 * 10;  // 10° per second

		 glm::mat4 Projection = glm::perspective(glm::radians(45.0f), 1.0f*screenWidth / screenHeight, 0.1f, 100.0f);
		 glm::mat4 View = glm::lookAt(glm::vec3(0.0, 0.0, -5.0), glm::vec3(0.0, 0.0, 0.0),
			 glm::vec3(0.0, 1.0, 0.0));
		 glm::mat4 Model = glm::translate(glm::mat4(1.0f), glm::vec3(0.0, 0.0, 0.0));
		 Model = glm::rotate(Model, glm::radians(angle), glm::vec3(0.0f, 1.0f, 0.0f));
		 glm::mat4 mvp_matrix = Projection * View * Model;

		 glUseProgram(programRenderID); 
		 glClearColor(0.0, 0.0, 1.0, 1.0);
		 glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		 // 1rst attribute buffer : vertices
		 glEnableVertexAttribArray(0);
		 glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer3d);
		 glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

		 // 2nd atrtibute: mvp matrix
		 glEnableVertexAttribArray(1);
		 glUniformMatrix4fv(1, 1, GL_FALSE, glm::value_ptr(mvp_matrix));

		 glDrawArrays(GL_POINTS, 0, 2048 * 1024);
		 glDisableVertexAttribArray(1);
		 glDisableVertexAttribArray(0);

		 SDL_GL_SwapWindow(window);

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
