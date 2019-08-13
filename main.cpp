#include <iostream>
#include <map>
#include <vector>
#include <chrono>
#include <string>
#include <CL/cl.h>
#include <cstdlib>
#include <cstdio>
#include <sys/time.h>
#include <iostream>

std::map<std::string, std::vector<int> > stopwatch_timings;
std::map<std::string, std::chrono::steady_clock::time_point> start_times;

std::chrono::steady_clock::time_point current_time() {
	return std::chrono::steady_clock::now();
}

void start_timing(const char* identifier)
{
	if (start_times.find(identifier) == start_times.end())
	{
		start_times.insert({ identifier, current_time() });
	} else
	{
		start_times[identifier] = current_time();
	}
}

void stop_timing(const char* identifier)
{
	std::chrono::steady_clock::time_point end = current_time();
	std::chrono::steady_clock::time_point start = start_times[identifier];
	int duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

	if (stopwatch_timings.find(identifier) == stopwatch_timings.end())
	{
		stopwatch_timings.insert({ identifier, std::vector<int>() });
	}

	stopwatch_timings[identifier].push_back(duration);
}

#define CNN_STOPWATCH(name) for(bool _start = true; (_start ? start_timing(name) : stop_timing(name)), _start; _start = false)

// includes

cl_int errNum;
cl_context context;
cl_command_queue commandQueue;
cl_program program;
cl_program programfloat4;

int min(int index, int f4);

const int COUNT2 = 119;
cl_float4 weightsfloat4[9] = {
	(cl_float4) {0.11711525917053223f,
		0.2533043324947357f,
		-0.1808098554611206f,
		-0.022508807480335236f},
	(cl_float4) {-0.6649995446205139f,
		0.09314073622226715f,
		-0.003124025883153081f,
		0.017212042585015297f},
	(cl_float4) {-0.00145575066562742f,
		-0.013053400442004204f,
		0.023538049310445786f,
		-0.00013999752991367131f},
	(cl_float4) {-0.007973596453666687f,
		-0.0006502429023385048f,
		0.0026328787207603455f,
		-0.5550334453582764f},
	(cl_float4) {0.3889226019382477f,
		-0.11613842099905014f,
		0.2719026505947113f,
		0.3205232620239258f},
	(cl_float4) {0.056865300983190536f,
		-0.11111960560083389f,
		0.22497615218162537f,
		-0.30552172660827637f},
	(cl_float4) {-0.2547873556613922f,
		0.05941327288746834f,
		-0.5109447836875916f,
		0.046159517019987106f},
	(cl_float4) {0.32608509063720703f,
		0.09823229908943176f,
		0.040836017578840256f,
		0.022866737097501755f},
	(cl_float4) {-0.033735278993844986f,
		-0.050256337970495224f,
		-0.025790197774767876f,
		0.042403850704431534f}};

#define LINEAR_3(i1, i2, i3, d2, d3) ((i1) * (d2) * (d3) + (i2) * (d3) + (i3))
#define LINEAR_4(i1, i2, i3, i4, d2, d3, d4) ((i1) * (d2) * (d3) * (d4) + (i2) * (d3) * (d4) + (i3) * (d4) + (i4))
//separable_conv2d_6_internal_1

void sepconv_serial_cpu()
{	
	const int H = 121;
    const int W = 161;				
    const int C_IN = 3;				
    const int C_OUT = 12;			
    const int W_OUT = 80;			
    const int SH = 2;				
    const int SW = 2;				
    const int KH = 3;				
    const int KW = 3;				
    const int DEPTH_MULTIPLIER = 4;	
    
    const int IN_OFFSET = 0;
    const int OUT_OFFSET = 0;
 
	for (int ix = 0; ix < H - KH + 1; ix += SH)
	{
		int x_out_1 = ix / SH;
		for (int jx = 0; jx < W - KW + 1; jx += SW)
		{
			int x_out_2 = jx / SW;
			for (int iw = 0; iw < KH; iw++)
			{
				int x_1 = ix + iw;
				for (int jw = 0; jw < KW; jw++)
				{
					int x_2 = jx + jw;
					for (int c = 0; c < C_IN; c++)
					{
						for (int m = 0; m < DEPTH_MULTIPLIER; m++)
						{
							int c_out = c * DEPTH_MULTIPLIER + m;
							std::cout << LINEAR_3(x_out_1, x_out_2, c_out,
									W_OUT, C_OUT) + OUT_OFFSET << " " << LINEAR_4(iw, jw, c, m, KW, C_IN, DEPTH_MULTIPLIER) << " " << LINEAR_3(x_1, x_2, c, W, C_IN) + IN_OFFSET << "\n";
							//out[LINEAR_3(x_out_1, x_out_2, c_out, W_OUT, C_OUT) + OUT_OFFSET] += weights[LINEAR_4(iw, jw, c, m, KW, C_IN, DEPTH_MULTIPLIER)] * in[LINEAR_3(x_1, x_2, c, W, C_IN) + IN_OFFSET];
						}
					}
				}
			}
		}
	}
}

int chooseDevice(cl_platform_id* platform_ids,
		const int chosendevice,
		cl_device_id* device_id)
{
	cl_device_id * device_ids;
	cl_uint numDevices; 
	errNum = clGetDeviceIDs(platform_ids[0], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);
	if (errNum != CL_SUCCESS)
	{
		printf("Error: Failed to create a device group!\n");
		return -1;
	}

	device_ids = (cl_device_id *)alloca(sizeof(cl_device_id) * numDevices);
	errNum = clGetDeviceIDs(platform_ids[0], CL_DEVICE_TYPE_ALL, numDevices, device_ids, NULL);

	std::size_t paramValueSize;

	errNum = clGetDeviceInfo(
			device_ids[chosendevice],
			CL_DEVICE_TYPE,
			0,
			NULL,
			&paramValueSize);
	if (errNum != CL_SUCCESS)
	{
		std::cerr << "Failed to find OpenCL device info " << "." << std::endl;
		return -1;
	}

	cl_device_type * info = (cl_device_type *)alloca(sizeof(cl_device_type) * paramValueSize);
	errNum = clGetDeviceInfo(
			device_ids[chosendevice],
			CL_DEVICE_TYPE,
			paramValueSize,
			info,
			NULL);
	if (errNum != CL_SUCCESS)
	{
		std::cerr << "Failed to find OpenCL device info " << "." << std::endl;
		return -1;
	}
	std::cout << "------------\n" << "CPU:" << CL_DEVICE_TYPE_CPU << "\n------------" <<std::endl;
	std::cout << "------------\n" << "GPU:" << CL_DEVICE_TYPE_GPU << "\n------------" <<std::endl;
	std::cout << "------------\n" << info[0] << "------------" <<std::endl;

	std::cout << "Device ids:\n";
	for (int i = 0; i < numDevices; i++)  {
		std::cout << i << ": " << device_ids[i] << std::endl;
	}
	*device_id = device_ids[chosendevice];
	return 0;
}

int initOcl()
{
	const char * kernelSourceString[COUNT2] = {
		"#define LINEAR_3(i1, i2, i3, d2, d3) ((i1) * (d2) * (d3) + (i2) * (d3) + (i3))\n",
		"#define LINEAR_4(i1, i2, i3, i4, d2, d3, d4) ((i1) * (d2) * (d3) * (d4) + (i2) * (d3) * (d4) + (i3) * (d4) + (i4))\n",
		"__kernel void conv(\n",
		"  __global const float *in,\n",
		"  __global float *out,\n",
		"  __global float *weights)\n",
		"{\n",
		"	const int H = 60;\n",
		"	const int W = 80; const int C_IN = 3; const int C_OUT = 8; const int H_OUT = 60; const int W_OUT = 80;\n",
		"	const int SH = 1; const int SW = 1;\n",
		"	const int KH = 1; const int KW = 1;\n",
		"	\n",
		"	const int IN_OFFSET = 0;\n",
		"	const int OUT_OFFSET = 0;\n",
		" \n",
		"	int x_out_1 = get_global_id(0);\n",
		"	int x_out_2 = get_global_id(1);\n",
		"	int ix = x_out_1 * SH;\n",
		"	int jx = x_out_2 * SW;\n",
		"	for (int iw = 0; iw < KH; iw++)\n",
		"	{\n",
		"		int x_1 = ix + iw;\n",
		"		for (int jw = 0; jw < KW; jw++)\n",
		"		{\n",
		"			int x_2 = jx + jw;\n",
		"			for (int kw = 0; kw < C_IN; kw++)\n",
		"			{\n",
		"				float x_in = in[LINEAR_3(x_1, x_2, kw, W, C_IN) + IN_OFFSET];\n",
		"				int lw;\n",
		"				for (; lw < C_OUT; lw++)\n",
		"				{\n",
		"					int w_index = LINEAR_4(iw, jw, kw, lw, KW, C_IN, C_OUT);\n",
		"					float w = weights[w_index];\n",
		"					int out_index = LINEAR_3(x_out_1, x_out_2, lw, W_OUT, C_OUT) + OUT_OFFSET;\n",
		"					out[out_index] += x_in * w;\n",
		"				}\n",
		"			}\n",
		"		}\n",
		"	}\n",
		"}\n",
		"	__kernel void float4conv(\n",
		"  __global const float *in,\n",
		"  __global float4 *out,\n",
		"  __global float4 *weights)\n",
		"{\n",
		"	const int H = 60;\n",
		"	const int W = 80; const int C_IN = 3; const int C_OUT = 8; const int H_OUT = 60; const int W_OUT = 80;\n",
		"	const int SH = 1; const int SW = 1;\n",
		"	const int KH = 1; const int KW = 1;\n",
		"	\n",
		"	const int IN_OFFSET = 0;\n",
		"	const int OUT_OFFSET = 0;\n",
		" \n",
		"	int x_out_1 = get_global_id(0);\n",
		"	int x_out_2 = get_global_id(1);\n",
		"	int ix = x_out_1 * SH;\n",
		"	int jx = x_out_2 * SW;\n",
		"	for (int iw = 0; iw < KH; iw++)\n",
		"	{\n",
		"		int x_1 = ix + iw;\n",
		"		for (int jw = 0; jw < KW; jw++)\n",
		"		{\n",
		"			int x_2 = jx + jw;\n",
		"			for (int kw = 0; kw < C_IN; kw++)\n",
		"			{\n",
		"				float4 x_in = (float4) in[LINEAR_3(x_1, x_2, kw, W, C_IN) + IN_OFFSET];\n",
		"				int lw;\n",
		"				for (lw = 0; lw < C_OUT - 3; lw += 4)\n",
		"				{\n",
		"					float4 y, x_out;\n",
		"					int w_index = LINEAR_4(iw, jw, kw, lw, KW, C_IN, C_OUT) /4;\n",
		"					y = x_in * weights[w_index];\n",
		"					int out_index = LINEAR_3(x_out_1, x_out_2, lw, W_OUT, C_OUT) + OUT_OFFSET /4;\n",
		"					out[out_index] = out[out_index] + y;\n",
		"				}\n",
		"			}\n",
		"		}\n",
		"	}\n",
		"}\n",
		"__kernel void sepconv_serial(\n",
		"  __global const float *in,\n",
		"  __global float *out,\n",
		"  __global float *weights)\n",
		"{\n",
		"	const int W = 161;				\n",
		"	const int C_IN = 3;				\n",
		"	const int C_OUT = 12;			\n",
		"	const int W_OUT = 80;			\n",
		"	const int SH = 2;				\n",
		"	const int SW = 2;				\n",
		"	const int KH = 3;				\n",
		"	const int KW = 3;				\n",
		"	const int DEPTH_MULTIPLIER = 4;	\n",
		"	\n",
		"	const int IN_OFFSET = 0;\n",
		"	const int OUT_OFFSET = 0;\n",
		"   \n",
		"	int x_out_1 = get_global_id(0);\n",
		"	int x_out_2 = get_global_id(1);\n",
		"   \n",
		"	int ix = x_out_1 * SH;\n",
		"	int jx = x_out_2 * SW;\n",
		"	for (int iw = 0; iw < KH; iw++)\n",
		"	{\n",
		"		int x_1 = ix + iw;\n",
		"		for (int jw = 0; jw < KW; jw++)\n",
		"		{\n",
		"			int x_2 = jx + jw;\n",
		"			for (int c = 0; c < C_IN; c++)\n",
		"			{\n",
		"				for (int m = 0; m < DEPTH_MULTIPLIER; m++)\n",
		"				{\n",
		"					int c_out = c * DEPTH_MULTIPLIER + m;\n",
		"					out[LINEAR_3(x_out_1, x_out_2, c_out, W_OUT, C_OUT) + OUT_OFFSET] += weights[LINEAR_4(iw, jw, c, m, KW, C_IN, DEPTH_MULTIPLIER)] * in[LINEAR_3(x_1, x_2, c, W, C_IN) + IN_OFFSET];\n",
		"				}\n",
		"			}\n",
		"		}\n",
		"	}\n",
		"}\n"
		};


	cl_uint dev_cnt = 0;
	clGetPlatformIDs(0, 0, &dev_cnt);

	cl_platform_id platform_ids[100];
	clGetPlatformIDs(dev_cnt, platform_ids, NULL);

	cl_device_id device_id;
	const int chosendevice = 0;
	int retval = 0;
	if ((retval = chooseDevice(platform_ids, chosendevice, &device_id))) {
			return(retval);
	}

	context = clCreateContext(0, 1, &device_id, NULL, NULL, &errNum);
	if (!context)
	{
		printf("Error: Failed to create a compute context!\n");
		return -1;
	}

	commandQueue = clCreateCommandQueue(context, device_id, 0, &errNum);
	if (!commandQueue)
	{
		printf("Error: Failed to create a command commands!\n");
		return -1;
	}

	program = clCreateProgramWithSource(context, COUNT2, (const char **) kernelSourceString, NULL, &errNum);


	if (!program)
	{
		printf("Error: Failed to create compute program! errnum: %d COUNT2: %d kernelSourceString: %lu\n", errNum, COUNT2, sizeof(kernelSourceString[0]));
		for (int i = 0; i < COUNT2; i++)
			if (kernelSourceString[i] == NULL)
				printf("%d is NULL.\n", i);
		return EXIT_FAILURE;
	}


	errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (errNum != CL_SUCCESS)
	{
		size_t len;
		char buffer[2048];
		printf("Error: Failed to build program executable! errnum: %d\n", errNum);
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
		printf("%s\n", buffer);
		exit(1);
	}
	return errNum;
}


int createKernel(cl_program  *program, cl_kernel *kernel, char kernelName[])
{
	*kernel = clCreateKernel(*program, kernelName, &errNum);
	if (!*kernel || errNum != CL_SUCCESS)
	{
		printf("Error: Failed to create compute kernel!\n");
		return -1;
	}
	return 0;
}

void cnn(cl_mem in_0, cl_mem weights, cl_mem out_0, size_t dim, cl_program * prog, char* name)
{
	{
		{
			cl_kernel layer;
			size_t localWorkSize[1] = {1};
			size_t globalWorkSize[1] = {dim};
			{

				createKernel(prog, &layer, name);

				if (in_0 == NULL) {
					printf("Given input memObject is null in %s.\n", name);
					exit(-1);
				}
				if (weights == NULL) {
					printf("Given weights memObject is null in %s.\n", name);
					exit(-1);
				}
				if (out_0 == NULL) {
					printf("Given output memObject is null in %s.\n", name);
					exit(-1);
				}
				errNum = clSetKernelArg(layer, 0, sizeof(cl_mem), (void *) &in_0);
				errNum |= clSetKernelArg(layer, 1, sizeof(cl_mem), (void *) &weights);
				errNum |= clSetKernelArg(layer, 2, sizeof(cl_mem), (void *) &out_0);

				if (errNum != CL_SUCCESS) {
					printf("Error setting Kernel Arguments in %s.\n", name);
					exit(-1);
				}
			}
			{
				// enqueue and wait for processing to end
				errNum = clEnqueueNDRangeKernel(commandQueue, layer, 1,
						NULL, globalWorkSize,
						localWorkSize, 0, NULL,
						NULL);
				if (errNum != 0) {
					printf("Error Status %d in clEnqueueNDRangeKerneltestkernel\n", errNum);
					exit(-1);
				}
				clFinish(commandQueue);
			}
		}
	}
}

void cnn(cl_mem in_0, cl_mem out_0, size_t dim, cl_program * prog, char* name)
{
	{
		{
			cl_kernel layer;
			size_t localWorkSize[1] = {1};
			size_t globalWorkSize[1] = {dim};
			{

				createKernel(prog, &layer, name);

				if (in_0 == NULL) {
					printf("Given input memObject is null in %s.\n", name);
					exit(-1);
				}
				if (out_0 == NULL) {
					printf("Given output memObject is null in %s.\n", name);
					exit(-1);
				}
				errNum = clSetKernelArg(layer, 0, sizeof(cl_mem), (void *) &in_0);
				errNum |= clSetKernelArg(layer, 1, sizeof(cl_mem), (void *) &out_0);

				if (errNum != CL_SUCCESS) {
					printf("Error setting Kernel Arguments in %s.\n", name);
					exit(-1);
				}
			}
			{
				// enqueue and wait for processing to end
				errNum = clEnqueueNDRangeKernel(commandQueue, layer, 1,
						NULL, globalWorkSize,
						localWorkSize, 0, NULL,
						NULL);
				if (errNum != 0) {
					printf("Error Status %d in clEnqueueNDRangeKerneltestkernel\n", errNum);
					exit(-1);
				}
				clFinish(commandQueue);
			}
		}
	}
}



int main()
{
	const size_t IN_DIM		= 58443;
	const size_t OUT_DIM		= 58000;
	const size_t WEIGHTS_DIM  = 108;

	const size_t IN_DIM4		= 58444 / 4;
	const size_t OUT_DIM4		= 58000 / 4;
	const size_t WEIGHTS_DIM4 = 108 / 4;

	cl_float in[IN_DIM];
	cl_float out[OUT_DIM];
	cl_float weights[WEIGHTS_DIM];

	cl_float4 in4[IN_DIM4]; 
	cl_float4 out4[OUT_DIM4];
	cl_float4 weights4[WEIGHTS_DIM4];

	for (int i = 0; i < IN_DIM; ++i) {
		in[i] = static_cast<cl_float>(i);
	}
	for (int i = 0; i < WEIGHTS_DIM; ++i) {
		weights[i] = static_cast<cl_float>(i);
	}

	for (int i = 0; i < IN_DIM4; i+=4) {
		in4[i/4] = (cl_float4) {static_cast<cl_float>(i), static_cast<cl_float>(i+1), static_cast<cl_float>(i+2), static_cast<cl_float>(i+3)};
	}
	for (int i = 0; i < OUT_DIM4; i+=4) {
		weights4[i/4] = (cl_float4) {static_cast<cl_float>(i), static_cast<cl_float>(i+1), static_cast<cl_float>(i+2), static_cast<cl_float>(i+3)};
	}

	if (initOcl()) {
		fprintf(stderr,"Failed InitOcl\n");
		exit(-1);
	}
	//initclmemobjects();

	for (int j = 0; j < 10; ++j) {
		cl_mem cl_in;
		cl_mem cl_weights;
		cl_mem cl_out;
		cl_in = clCreateBuffer(context,
				CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
				IN_DIM * sizeof(cl_float), in, &errNum);
		cl_weights = clCreateBuffer(context,
				CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
				WEIGHTS_DIM * sizeof(cl_float), weights, &errNum);
		cl_out = clCreateBuffer(context,
				CL_MEM_READ_WRITE,
				OUT_DIM * sizeof(cl_float), NULL, &errNum);
		// reset out
		for (size_t i = 0; i < OUT_DIM; i++) {
			out[i] = 0.0f;
		}
		char name[] = "sepconv_serial";

		// run function
		CNN_STOPWATCH(std::to_string(IN_DIM).c_str()) {
			cnn(cl_in, cl_weights, cl_out, IN_DIM, &program, name);
		}
		// get memory from gpu
		errNum = clEnqueueReadBuffer(commandQueue,
				cl_out,
				CL_TRUE,
				0,
				OUT_DIM * sizeof(float),
				out,
				0,
				NULL,
				NULL);
		if (errNum != 0) {
			printf("Error Status %d in clEnqueueReadBuffer\n", errNum);
			exit(-1);
		}
		clReleaseMemObject(cl_in);
		clReleaseMemObject(cl_out);
	}
		//printf("F4IN_DIM: %4.lu TIME: %7.d TIME_PER_ELEMENT: %5.lu ", IN_DIM, min(IN_DIM, 1), min(IN_DIM, 1) / IN_DIM);
		printf("__IN_DIM: %4.lu TIME: %7.d TIME_PER_ELEMENT: %5.lu\n", IN_DIM, min(IN_DIM, 0), min(IN_DIM, 0) / IN_DIM);

	//printf("time: %d nano seconds\nsize: %d\naverage: %d nano seconds/run\n", total_elapsed, IN_DIM - 1, total_elapsed/IN_DIM);
	return 0;
}

int min(int index, int f4) {
	int min {INT32_MAX};
	int avg = 0;
	std::string s;
	if (f4)
		s = std::to_string(index) + " float4";
	else 
		s = std::to_string(index);
	const char * s1 = s.c_str();
	for (size_t i = 0; i < stopwatch_timings[s1].size(); i+=1) {
		min = min <= stopwatch_timings[s1][i] ? min : stopwatch_timings[s1][i];
	}
	for (size_t i = 0; i < stopwatch_timings[s1].size(); i+=1) {
		avg += stopwatch_timings[s1][i];
	}
	avg /= stopwatch_timings[s1].size();
	return avg;
	return min;
}

