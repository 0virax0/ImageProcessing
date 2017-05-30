#include "clBase.h"
#include "glBase.h"
char* THIS_FOLDER = "C:\\Users\\filip\\Desktop\\ImageProcessing";

using namespace std;

class clWrapper;
clWrapper* opencl;
cl_context context;
cl_int err;
cl_command_queue commandQueue;
glWrapper* opengl;

template<typename T>
const char* checkCodeStr(T input)
{
	int errorCode = (int)input;
	switch (errorCode)
	{
	case CL_DEVICE_NOT_FOUND:
		return "CL_DEVICE_NOT_FOUND";
	case CL_DEVICE_NOT_AVAILABLE:
		return "CL_DEVICE_NOT_AVAILABLE";
	case CL_COMPILER_NOT_AVAILABLE:
		return "CL_COMPILER_NOT_AVAILABLE";
	case CL_MEM_OBJECT_ALLOCATION_FAILURE:
		return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
	case CL_OUT_OF_RESOURCES:
		return "CL_OUT_OF_RESOURCES";
	case CL_OUT_OF_HOST_MEMORY:
		return "CL_OUT_OF_HOST_MEMORY";
	case CL_PROFILING_INFO_NOT_AVAILABLE:
		return "CL_PROFILING_INFO_NOT_AVAILABLE";
	case CL_MEM_COPY_OVERLAP:
		return "CL_MEM_COPY_OVERLAP";
	case CL_IMAGE_FORMAT_MISMATCH:
		return "CL_IMAGE_FORMAT_MISMATCH";
	case CL_IMAGE_FORMAT_NOT_SUPPORTED:
		return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
	case CL_BUILD_PROGRAM_FAILURE:
		return "CL_BUILD_PROGRAM_FAILURE";
	case CL_MAP_FAILURE:
		return "CL_MAP_FAILURE";
	case CL_MISALIGNED_SUB_BUFFER_OFFSET:
		return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
	case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
		return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
	case CL_INVALID_VALUE:
		return "CL_INVALID_VALUE";
	case CL_INVALID_DEVICE_TYPE:
		return "CL_INVALID_DEVICE_TYPE";
	case CL_INVALID_PLATFORM:
		return "CL_INVALID_PLATFORM";
	case CL_INVALID_DEVICE:
		return "CL_INVALID_DEVICE";
	case CL_INVALID_CONTEXT:
		return "CL_INVALID_CONTEXT";
	case CL_INVALID_QUEUE_PROPERTIES:
		return "CL_INVALID_QUEUE_PROPERTIES";
	case CL_INVALID_COMMAND_QUEUE:
		return "CL_INVALID_COMMAND_QUEUE";
	case CL_INVALID_HOST_PTR:
		return "CL_INVALID_HOST_PTR";
	case CL_INVALID_MEM_OBJECT:
		return "CL_INVALID_MEM_OBJECT";
	case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
		return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
	case CL_INVALID_IMAGE_SIZE:
		return "CL_INVALID_IMAGE_SIZE";
	case CL_INVALID_SAMPLER:
		return "CL_INVALID_SAMPLER";
	case CL_INVALID_BINARY:
		return "CL_INVALID_BINARY";
	case CL_INVALID_BUILD_OPTIONS:
		return "CL_INVALID_BUILD_OPTIONS";
	case CL_INVALID_PROGRAM:
		return "CL_INVALID_PROGRAM";
	case CL_INVALID_PROGRAM_EXECUTABLE:
		return "CL_INVALID_PROGRAM_EXECUTABLE";
	case CL_INVALID_KERNEL_NAME:
		return "CL_INVALID_KERNEL_NAME";
	case CL_INVALID_KERNEL_DEFINITION:
		return "CL_INVALID_KERNEL_DEFINITION";
	case CL_INVALID_KERNEL:
		return "CL_INVALID_KERNEL";
	case CL_INVALID_ARG_INDEX:
		return "CL_INVALID_ARG_INDEX";
	case CL_INVALID_ARG_VALUE:
		return "CL_INVALID_ARG_VALUE";
	case CL_INVALID_ARG_SIZE:
		return "CL_INVALID_ARG_SIZE";
	case CL_INVALID_KERNEL_ARGS:
		return "CL_INVALID_KERNEL_ARGS";
	case CL_INVALID_WORK_DIMENSION:
		return "CL_INVALID_WORK_DIMENSION";
	case CL_INVALID_WORK_GROUP_SIZE:
		return "CL_INVALID_WORK_GROUP_SIZE";
	case CL_INVALID_WORK_ITEM_SIZE:
		return "CL_INVALID_WORK_ITEM_SIZE";
	case CL_INVALID_GLOBAL_OFFSET:
		return "CL_INVALID_GLOBAL_OFFSET";
	case CL_INVALID_EVENT_WAIT_LIST:
		return "CL_INVALID_EVENT_WAIT_LIST";
	case CL_INVALID_EVENT:
		return "CL_INVALID_EVENT";
	case CL_INVALID_OPERATION:
		return "CL_INVALID_OPERATION";
	case CL_INVALID_GL_OBJECT:
		return "CL_INVALID_GL_OBJECT";
	case CL_INVALID_BUFFER_SIZE:
		return "CL_INVALID_BUFFER_SIZE";
	case CL_INVALID_MIP_LEVEL:
		return "CL_INVALID_MIP_LEVEL";
	case CL_INVALID_GLOBAL_WORK_SIZE:
		return "CL_INVALID_GLOBAL_WORK_SIZE";
	default:
		return "unknown error code";
	}
}

template<typename T>
int printErr(T input, string specs) {
	int errorCode = (int)input;
	std::string s = checkCodeStr(input);
	if (input != CL_SUCCESS) {
		std::cout << s << ": " << specs << endl;
		return FAILURE;
	}
	return SUCCESS;
}

#pragma region class destroyable
	template <typename T>
	destroyable<T>::destroyable(T* p, RAIIscope* scope) : reference(p) {	
		if (scope != NULL) {
			scope->localRAII->arr.push_back(this);
			localRAIIIndex = scope->localRAII->arr.size() - 1;
		}	
	}
	template <typename T>
	void destroyable<T>::destroy() {
		selfDestroy(reference);
	}

	template<>	//member specialization
	int destroyable<void*>::selfDestroy(void** obj) {
		cout << "free dynamic allocated memory" << endl;
		free(*obj);
		(*obj) = NULL;
		return 1;
	}
	template destroyable<void*>;	//forced template instanciation

	template<>
	int destroyable<cl_mem>::selfDestroy(cl_mem* obj) {
		cout << "free cl_mem" << endl;
		return printErr(clReleaseMemObject(*obj), "free failed");	
	}
	template destroyable<cl_mem>;	//forced

	template<>
	int destroyable<cl_kernel>::selfDestroy(cl_kernel* kernel) {
		cout << "free cl_kernel" << endl;
		return printErr(clReleaseKernel(*kernel), "free failed");
	}
	template destroyable<cl_kernel>;	//forced

	template<>
	int destroyable<GLuint>::selfDestroy(GLuint* buff) {
		cout << "free gl_buffer" << endl;
		glDeleteBuffers(1, buff);
		return 1;
	}
	template destroyable<GLuint>;	//forced

	//general case
	template <typename T> int destroyable<T>::selfDestroy(T* obj) {
		cout << "free generic object" << endl;
		delete *obj;
		(*obj) = NULL;
		return 1;
	}
	//destructor
	template <typename T> destroyable<T>::~destroyable() {
		selfDestroy(reference);

	}
	
#pragma endregion
#pragma region class destroyer
	destroyer::destroyer(){ 

	}
	destroyer::~destroyer() {
		//destroy all objects
		for (std::vector<innerDestroyable*>::reverse_iterator it = arr.rbegin(); it != arr.rend(); ++it) {			
			//if the scope hasn't destroyed it
			innerDestroyable* ptr = *(it);
			if (ptr != NULL)
				ptr->destroy();	
		}
	}
#pragma endregion
#pragma region class RAIIscope
	RAIIscope::RAIIscope() {
		localRAII = new destroyer();
	}
	RAIIscope::~RAIIscope() {
		delete localRAII;
	}
#pragma endregion
int convertToString(const char *filename, char** s)
{
	size_t size;
	char*  str;
	std::fstream f(filename, (std::fstream::in | std::fstream::binary));

	if (f.is_open())
	{
		size_t fileSize;
		f.seekg(0, std::fstream::end);
		size = fileSize = (size_t)f.tellg();
		f.seekg(0, std::fstream::beg);
		str = new char[size + 1];
		if (!str)
		{
			f.close();
			return FAILURE;
		}

		f.read(str, fileSize);
		f.close();
		str[size] = '\0';
		*s = str;
		
		return SUCCESS;
	}
	std::cout << "Error: failed to open file\n:" << filename << endl;
	return FAILURE;
}
GLuint LoadShaders(const char * vertex_file_path, const char * fragment_file_path) {

	// Create the shaders
	GLuint VertexShaderID = glCreateShader(GL_VERTEX_SHADER);
	GLuint FragmentShaderID = glCreateShader(GL_FRAGMENT_SHADER);

	// Read the Vertex Shader code from the file
	char* VertexShaderCode=NULL;
	convertToString(vertex_file_path, &VertexShaderCode);

	// Read the Fragment Shader code from the file
	char* FragmentShaderCode=NULL;
	convertToString(fragment_file_path, &FragmentShaderCode);

	GLint Result = GL_FALSE;
	int InfoLogLength;

	// Compile Vertex Shader
	printf("Compiling shader : %s\n", vertex_file_path);
	char const * VertexSourcePointer = VertexShaderCode;
	glShaderSource(VertexShaderID, 1, &VertexSourcePointer, NULL);
	glCompileShader(VertexShaderID);

	// Check Vertex Shader
	glGetShaderiv(VertexShaderID, GL_COMPILE_STATUS, &Result);
	glGetShaderiv(VertexShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
	if (InfoLogLength > 0) {
		std::vector<char> VertexShaderErrorMessage(InfoLogLength + 1);
		glGetShaderInfoLog(VertexShaderID, InfoLogLength, NULL, &VertexShaderErrorMessage[0]);
		printf("%s\n", &VertexShaderErrorMessage[0]);
	}


	// Compile Fragment Shader
	printf("Compiling shader : %s\n", fragment_file_path);
	char const * FragmentSourcePointer = FragmentShaderCode;
	glShaderSource(FragmentShaderID, 1, &FragmentSourcePointer, NULL);
	glCompileShader(FragmentShaderID);

	// Check Fragment Shader
	glGetShaderiv(FragmentShaderID, GL_COMPILE_STATUS, &Result);
	glGetShaderiv(FragmentShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
	if (InfoLogLength > 0) {
		std::vector<char> FragmentShaderErrorMessage(InfoLogLength + 1);
		glGetShaderInfoLog(FragmentShaderID, InfoLogLength, NULL, &FragmentShaderErrorMessage[0]);
		printf("%s\n", &FragmentShaderErrorMessage[0]);
	}


	// Link the program
	printf("Linking program\n");
	GLuint ProgramID = glCreateProgram();
	glAttachShader(ProgramID, VertexShaderID);
	glAttachShader(ProgramID, FragmentShaderID);
	glLinkProgram(ProgramID);

	// Check the program
	glGetProgramiv(ProgramID, GL_LINK_STATUS, &Result);
	glGetProgramiv(ProgramID, GL_INFO_LOG_LENGTH, &InfoLogLength);
	if (InfoLogLength > 0) {
		std::vector<char> ProgramErrorMessage(InfoLogLength + 1);
		glGetProgramInfoLog(ProgramID, InfoLogLength, NULL, &ProgramErrorMessage[0]);
		printf("%s\n", &ProgramErrorMessage[0]);
	}


	glDetachShader(ProgramID, VertexShaderID);
	glDetachShader(ProgramID, FragmentShaderID);

	glDeleteShader(VertexShaderID);
	glDeleteShader(FragmentShaderID);

	return ProgramID;
}

#pragma region class image
		image::image() : RAIIscope(){

			format.image_channel_order = CL_RGBA;
			format.image_channel_data_type = CL_UNORM_INT8;
		}

		int image::loadImage(const char *filename) {
			tjhandle _jpegDecompressor = tjInitDecompress();
			unsigned char* _compressedImage; //!< _compressedImage from stream
			size_t size;
			std::fstream file(filename, (std::fstream::in | std::fstream::binary));

			if (file.is_open())
			{
				size_t fileSize;
				file.seekg(0, std::fstream::end);
				size = fileSize = (size_t)file.tellg();
				file.seekg(0, std::fstream::beg);

				_compressedImage = new unsigned char[size];
				file.read((char*)_compressedImage, size);
				destroyable<void*>wr((void**)&_compressedImage, NULL);

				//decompress header		
				int h, w;
				tjDecompressHeader2(_jpegDecompressor, _compressedImage, size, &w, &h, &jpegSubsamp);
				width = (size_t)((unsigned)w);
				height = (size_t)((unsigned)h);
				size = COLOR_COMPONENTS * width * height;

				//create image clmem
				inputImage = clCreateImage2D(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, &format, width, height, 0, NULL, &err);
				printErr(err, "imageIn cl_mem failed");
				new destroyable<cl_mem>(&inputImage, this);

				//create gl- cl shared buffer
				outputImage = opengl->createGLtexture(width, height, CL_MEM_READ_WRITE, &outTextureID);
				//imageOutBuffer=(unsigned char*)malloc(height*width*sizeof(unsigned char)*4);
				//outputImage=clCreateImage2D(context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, &format, width, height, 0, (void* )imageOutBuffer, 0);
				new destroyable<cl_mem>(&outputImage, this);

				//mapping
				size_t origin[3] = { 0, 0, 0 };
				size_t region[3] = { width, height, 1 };
				size_t rowPitch;
				imageBuffer = (unsigned char*)clEnqueueMapImage(commandQueue, inputImage, CL_TRUE, CL_MAP_WRITE,
					origin, region, &rowPitch, NULL, 0, NULL, NULL, &err);
				printErr(err, "image initial mapping failed");

				//fill with decompressed data
				tjDecompress2(_jpegDecompressor, _compressedImage, size, imageBuffer, width, 0, height, TJPF_RGBX, TJFLAG_FASTDCT);

				//unmap
				clEnqueueUnmapMemObject(commandQueue, inputImage, imageBuffer, 0, NULL, NULL);

				return SUCCESS;
			}
			std::cout << "Error: failed to open file: " << filename << endl;
			tjDestroy(_jpegDecompressor);
			file.close();
			return FAILURE;
		}
		int image::fetchOutImage() {	
			//memory for imageOutBuffer has to be allocated(now is not)
			size_t origin[3] = { 0, 0, 0 };
			size_t region[3] = { width, height, 1 };

			return printErr(clEnqueueReadImage(commandQueue, outputImage, CL_TRUE, origin, region, 0, 0, (void *)imageOutBuffer, 0, NULL, NULL), "getting outbuffer failed");
		}
#pragma endregion

int saveImage(const char *filename, image& img) {
	const int JPEG_QUALITY = 75;
	const int COLOR_COMPONENTS = 4;
	long unsigned int _jpegSize = 0;
	unsigned char* _convertedImage = NULL; //!< Memory is allocated by tjCompress2 if _jpegSize == 0
	unsigned char* _compressedImage = NULL;

	tjhandle _jpegCompressor = tjInitCompress();

	if (tjCompress2(_jpegCompressor, img.imageOutBuffer, img.width, 0, img.height, TJPF_RGBX,
		&_compressedImage, &_jpegSize, img.jpegSubsamp, JPEG_QUALITY,
		TJFLAG_FASTDCT) == -1) {
		tjGetErrorStr();
		return 0;
	};
	tjDestroy(_jpegCompressor);

	//write to file
	ofstream offFile;
	offFile.open(filename);
	std::cout << _jpegSize;
	std::cout << " ";
	offFile.write((char *)_compressedImage, _jpegSize);
	offFile.close();

	tjFree(_compressedImage);
	return 1;
}

typedef struct kernel_nonPtr_Args{
	int width;
	int height;
};

class clWrapper : RAIIscope {
public:
	cl_platform_id platform;	//the chosen platform
	cl_device_id *devices;
	cl_program program;
	image* img;
	cl_kernel kernel;
	cl_mem imgKernel;
	cl_mem imgKernel1;
public:
	//constructor
	clWrapper() : RAIIscope() {};

	/*Step1: Getting platforms and choose an available one.*/
	int getPlatform() {
		cl_uint numPlatforms;	//the NO. of platforms
		printErr(clGetPlatformIDs(0, NULL, &numPlatforms),"Error: Getting platforms!");

		/* choose the first available platform. */
		if (numPlatforms > 0)
		{
			cl_platform_id* platforms = (cl_platform_id*)malloc(numPlatforms * sizeof(cl_platform_id));
			printErr(clGetPlatformIDs(numPlatforms, platforms, NULL),"failed to get platform IDs");
			platform = platforms[0];
			free(platforms);
		}
		return 1;
	}
	
	/*Step 2:Query the platform and choose the first GPU device if has one.Otherwise use the CPU as device.*/
	int getDevice() {
		cl_uint				numDevices = 0;
		printErr(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices),"failed to get device IDs of gpu");
		if (numDevices == 0)	//no GPU available.
		{
			std::cout << "No GPU device available." << endl;
			std::cout << "Choose CPU as default device." << endl;
			printErr(clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 0, NULL, &numDevices),"failed to get device IDs of cpu");
			devices = (cl_device_id*)malloc(numDevices * sizeof(cl_device_id));
			clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, numDevices, devices, NULL);
			if (devices == NULL)cout << "no device found";
		}
		else
		{
			devices = (cl_device_id*)malloc(numDevices * sizeof(cl_device_id));
			printErr(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices, NULL), "failed to get device IDs of gpu");
		}
		new destroyable<void*>((void**)&devices, this);
		return 1;
	}

	/*Step 3: Create context.*/
	int createContext() {
		opengl = new glWrapper();
		opengl->initContext(platform, devices);
		return 1;
	}

	/*Step 4: Creating command queue associate with the context.*/
	int createCommandQueue() {
		commandQueue = clCreateCommandQueue(context, devices[0], 0, NULL);
		return 1;
	}

	/*Step 5: Create program object */
	int createProgram() {
		char str1[100]; strcpy(str1, THIS_FOLDER); strcat(str1, "\\bin\\x86_64\\Debug\\HelloWorld_Kernel.cl");
		const char *filename = str1;
		char* sourceStr = NULL;
		convertToString(filename, &sourceStr);
		destroyable<void*>wr((void**)&sourceStr, NULL);	//destroy within scope
		const char* source = sourceStr;
		size_t sourceSize[] = { strlen(source) }; 

		program = clCreateProgramWithSource(context, 1, &source, sourceSize, NULL);

		return 1;
	}

	/*Step 6: Build program. */
	int buildProgram() {
		if (clBuildProgram(program, 1, devices, NULL, NULL, NULL) != CL_SUCCESS)
		{
			char * buildLog = NULL;
			size_t buildLogSize = 10000;

			buildLog = (char*)malloc(buildLogSize);		
			destroyable<void*>wr((void**)&buildLog, NULL);
			memset(buildLog, 0, buildLogSize);
			printErr(clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, buildLogSize, buildLog, &buildLogSize),"unable to print kernel compilation errors");

			cout << buildLog;
		}
		return 1;
	}

	/*Step 7: Initial input,output for the host and create memory objects for the kernel*/
	int loadImage() {
		//load image To GPU
		img = new image();
		new destroyable<image*>(&img, opencl);
		char str1[100]; strcpy(str1, THIS_FOLDER); strcat(str1, "\\bin\\x86_64\\Debug\\imgTest.jpg");
		img->loadImage(str1);
		if (img->size == 0) return 0;
		return 1;
	}

	/*Step 8: Create kernel object*/
	int createKernel() {
		kernel = clCreateKernel(program, "filter", &err);
		printErr(err, "kernel creation failed");  
		new destroyable<cl_kernel>(&kernel, opencl);
		return 1;
	}

	/*Step 9: Sets Kernel arguments.*/
	int setKargs() {
		//create a non ptr args BUffer
		kernel_nonPtr_Args Barg{ img->width, img->height };
		cl_mem argBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(kernel_nonPtr_Args), (void *)&Barg, &err);
		printErr(err, "argBuffer failed");

		float imgKernelPtr[] = {   
			1,  0,  -1,
			2,  0,  -2,
			1,  0,  -1
		};
		imgKernel = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 9 * sizeof(float), (void *)imgKernelPtr, &err);
		printErr(err, "imgkernel failed");  
		new destroyable<cl_mem>(&imgKernel, opencl); 

		float imgKernelPtr1[] = {    
			-1, -2, -1,
			 0,  0,  0,
			 1,  2,  1
		};
		imgKernel1 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 9 * sizeof(float), (void *)imgKernelPtr1, &err);
		printErr(err, "imgkernel failed");  
		new destroyable<cl_mem>(&imgKernel1, opencl);
		
		printErr(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&argBuffer), "failed set argBuffer arg");
		printErr(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&imgKernel), "failed set imgkernel arg");
		printErr(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&imgKernel1), "failed set imgkernel1 arg");
		printErr(clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&img->inputImage), "failed set input arg");
		printErr(clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&img->outputImage), "failed set output arg");
		 
		return 1;
	}

	/*Step 10: Running the kernel.*/
	int startKernel() {
		size_t global_work_size[] = { img->width, img->height, 0 };
		cout << "start enqueue kernel" << endl;
		for (int i = 0; i < 1;i++)
			printErr(clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL), "failed ndrange enqueue");
		return 1;
	}

	//final Step 11: write to screen
	int getToScreen() {
		return opengl->glOps(img);
		
	}

	/*Step 12: Clean the resources.*/
	int cleanUp() {
		cout << "free glWrapper" << endl;
		opengl->Cleanup();
		cout << "free cl_command_queue" << endl;
		printErr(clReleaseCommandQueue(commandQueue), "free commandQueue failed");
		cout << "free cl_program" << endl;
		printErr(clReleaseProgram(program), "free program failed");
		cout << "free cl_context" << endl;
		 printErr(clReleaseContext(context), "free context failed");
		 return 1;
	}

	//destructor
	~clWrapper() {
		cleanUp();
	}

};

int main(int argc, char* argv[])
{
	opencl = new clWrapper();

	opencl->getPlatform();
	opencl->getDevice();
	opencl->createContext();
	opencl->createCommandQueue();
	opencl->createProgram();
	opencl->buildProgram();
	opencl->loadImage();
	opencl->createKernel();
	opencl->setKargs();
	opencl->startKernel(); 
	//opencl->img->fetchOutImage();
	opencl->getToScreen();

	clFinish(commandQueue);
	std::cout << "Passed!\n";
	std::system("pause");

	delete opencl;
	return SUCCESS;
}
