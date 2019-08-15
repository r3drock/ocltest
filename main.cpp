#define LINEAR_3(i1, i2, i3, d2, d3) ((i1) * (d2) * (d3) + (i2) * (d3) + (i3))
#define LINEAR_4(i1, i2, i3, i4, d2, d3, d4) ((i1) * (d2) * (d3) * (d4) + (i2) * (d3) * (d4) + (i3) * (d4) + (i4))
#ifdef __clang__
#define FORCE_INLINE __attribute__((always_inline))
#elif __gnuc__
#define FORCE_INLINE __attribute__((always_inline)) inline
#elif _MSC_VER
#define FORCE_INLINE __forceinline
#else
#define FORCE_INLINE /* inline */
#endif

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
#include <assert.h>
#include <string.h>

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

FORCE_INLINE void COPY(void *dest, const void *src, int n, int method)
{
	if (method == 0)
	{        
		memcpy(dest, src, n);
	}
	else
	{
		assert(0);
	}
}
#define CNN_STOPWATCH(name) for(bool _start = true; (_start ? start_timing(name) : stop_timing(name)), _start; _start = false)

// includes

cl_int errNum;
cl_context context;
cl_command_queue commandQueue;
cl_program program;
cl_program programfloat4;

int min(char* s1, int f4);
int min(int index, int f4);

void conv2dcpu(float* in_, float* out_, float* weights_)
{
	// OpConvolution2D
	const int W = 80; const int C_IN = 3; const int C_OUT = 8; const int H_OUT = 60; const int W_OUT = 80;
	const int SH = 1; const int SW = 1;
	const int KH = 1; const int KW = 1;

	for (int x_out_1 = 0; x_out_1 < H_OUT; x_out_1++)
	{
		int ix = x_out_1 * SH;
		for (int x_out_2 = 0; x_out_2 < W_OUT; x_out_2++)
		{
			int jx = x_out_2 * SW;
			alignas(16) float out_tmp[C_OUT];
			COPY(&out_tmp, &out_[LINEAR_3(x_out_1, x_out_2, 0, W_OUT, C_OUT)], C_OUT * sizeof(float), 0);
			int tempout = LINEAR_3(x_out_1, x_out_2, 0, W_OUT, C_OUT);
			for (int iw = 0; iw < KH; iw++)
			{
				int x_1 = ix + iw;
				for (int jw = 0; jw < KW; jw++)
				{
					int x_2 = jx + jw;
					for (int kw = 0; kw < C_IN; kw++)
					{
						int tempin = LINEAR_3(x_1, x_2, kw, W, C_IN);
						__m128 x_in = _mm_load_ps1(&in_[tempin]);
						for (int lw = 0; lw < C_OUT; lw += 4)
						{
							int w_index = LINEAR_4(iw, jw, kw, lw, KW, C_IN, C_OUT);
							//printf("cpu out: %d in: %d weights: %d\n", tempout + lw, tempin, w_index);
							//						    printf("in[%d]: %f * weights[%d]: %f = %f\n", tempin,
							//in_[tempin], w_index, weights_[w_index],
							//in_[tempin] * weights_[w_index]);

							__m128 w = _mm_load_ps(&weights_[w_index]);
							__m128 y = _mm_mul_ps(x_in, w);
							__m128 x_out = _mm_load_ps(&out_tmp[lw]);
							x_out = _mm_add_ps(x_out, y);
							_mm_store_ps(&out_tmp[lw], x_out);
						}
					}
				}
			}
			COPY(&out_[LINEAR_3(x_out_1, x_out_2, 0, W_OUT, C_OUT)], &out_tmp, C_OUT * sizeof(float), 0);
		}
	}
}

const int COUNT2 = 120;
float conv2d_11_internal_1_W[] = {1.2337225675582886f,-0.5368690490722656f,0.7839080691337585f,-1.8163790702819824f,0.9831442832946777f,-0.17836132645606995f,-0.41197478771209717f,0.8647937774658203f,-0.019595712423324585f,0.2376221865415573f,-1.819101333618164f,-0.6625561118125916f,0.13829778134822845f,2.1014888286590576f,-0.6704091429710388f,-0.6757892966270447f,-0.43060117959976196f,-0.9187846183776855f,-1.4585297107696533f,-1.9717609882354736f,0.19287244975566864f,2.419645309448242f,1.6170377731323242f,0.49752122163772583f};
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
		"  __global const float *weights,\n",
		"  __global float *out)\n",
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
		"				float x_in = in[LINEAR_3(x_1, x_2, kw, W, C_IN)];\n",
		"				for (int lw = 0; lw < C_OUT; lw++)\n",
		"				{\n",
		"					int w_index = LINEAR_4(iw, jw, kw, lw, KW, C_IN, C_OUT);\n",
		"					float w = weights[w_index];\n",
		"					int out_index = LINEAR_3(x_out_1, x_out_2, lw, W_OUT, C_OUT);\n",
		"					out[out_index] += x_in * w;\n",
		"				}\n",
		"			}\n",
		"		}\n",
		"	}\n",
		"}\n",
		"	__kernel void float4conv(\n",
		"  __global const float *in,\n",
		"  __global const float4 *weights,\n",
		"  __global float4 *out)\n",
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
		"				float4 x_in = (float4) in[LINEAR_3(x_1, x_2, kw, W, C_IN)];\n",
		"				int lw;\n",
		"				for (lw = 0; lw < C_OUT - 3; lw += 4)\n",
		"				{\n",
		"					float4 y, x_out;\n",
		"					int w_index = (LINEAR_4(iw, jw, kw, lw, KW, C_IN, C_OUT)) /4;\n",
		"					y = x_in * weights[w_index];\n",
		"					int out_index = (LINEAR_3(x_out_1, x_out_2, lw, W_OUT, C_OUT)) /4;\n",
		"					/*printf(\"gpu(%d,%d) out[%05d] : %.1f in[%05d] weights[%02d] in[%d]: %f * weights[%d]: %f = %f\\n\",",
		"								x_out_1, x_out_2, out_index, out[out_index].x, LINEAR_3(x_1, x_2, kw, W, C_IN), w_index, LINEAR_3(x_1, x_2, kw, W, C_IN),\n",
		"								x_in.x, w_index, weights[w_index].x,\n"
			"								y.x);*/\n"
			"					out[out_index] = out[out_index] + y;\n",
		"				}\n",
		"			}\n",
		"		}\n",
		"	}\n",
		"}\n",
		"__kernel void sepconv_serial(\n",
		"  __global const float *in,\n",
		"  __global float *weights,\n",
		"  __global float *out)\n",
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
			const size_t localWorkSize[2] = {1,1};
			const size_t globalWorkSize[2] = {60, 80};
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
				errNum = clEnqueueNDRangeKernel(commandQueue, layer, 2,
						NULL, globalWorkSize, localWorkSize,
						0, NULL, NULL);
				if (errNum != 0) {
					printf("Error Status %d in clEnqueueNDRangeKerneltestkernel\n", errNum);
					exit(-1);
				}
				clFinish(commandQueue);
			}
		}
	}
}


void printvalues(cl_float* arr, size_t len) {
	printf("len: %lu", len);
	for (size_t i = 0; i < len; ++i) {
		if ( i % 8 == 0)
			printf("\n");
		printf("%.1f ", arr[i]);
	}
	printf("\n\n");
}
void printvalues(cl_float4* arr, size_t len) {
	printf("len: %lu", len);
	for (size_t i = 0; i < len; ++i) {
		if ( i % 2 == 0)
			printf("\n");
		printf("%.1f ", arr[i].x);
		printf("%.1f ", arr[i].y);
		printf("%.1f ", arr[i].z);
		printf("%.1f ", arr[i].w);
	}
	printf("\n\n");
}


int main()
{
	const size_t IN_DIM		= 14400;
	const size_t OUT_DIM		= 38400;
	const size_t WEIGHTS_DIM  = sizeof(conv2d_11_internal_1_W) / sizeof(conv2d_11_internal_1_W[0]);

	if (IN_DIM % 4 != 0 || OUT_DIM % 4 != 0 || WEIGHTS_DIM % 4 != 0) {
		fprintf(stderr, "wrong Dimensions\n");
		exit(1);
	}

	const size_t OUT_DIM4		= OUT_DIM / 4;
	const size_t WEIGHTS_DIM4 = WEIGHTS_DIM / 4;

	cl_float in[IN_DIM];
	cl_float out[OUT_DIM];
	cl_float weights[WEIGHTS_DIM];

	cl_float4 out4[OUT_DIM4];
	cl_float4 weights4[WEIGHTS_DIM4];

	for (int i = 0; i < IN_DIM; ++i) {
		in[i] = static_cast<cl_float>(i);
	}
	for (int i = 0; i < WEIGHTS_DIM; ++i) {
		weights[i] = static_cast<cl_float>(i + 1);
	}

	for (int i = 0; i < OUT_DIM4; i+=4) {
		weights4[i/4] = (cl_float4) {static_cast<cl_float>(i + 1), static_cast<cl_float>(i+2), static_cast<cl_float>(i+3), static_cast<cl_float>(i+4)};
	}

	if (initOcl()) {
		fprintf(stderr,"Failed InitOcl\n");
		exit(-1);
	}
	char t[] = "cpu__";
	char* name = t;
	CNN_STOPWATCH(name) {
		for (int k = 0; k < 1; ++k) {
			for (size_t i = 0; i < OUT_DIM; i++) {
				out[i] = (cl_float) 0; 
			}
			conv2dcpu(in, out, weights);
		}
	}
	printf("CPU __IN_DIM: %4.lu TIME: %7.d TIME_PER_ELEMENT: %5.lu\n", IN_DIM, min(name, 0), min(name, 0) / IN_DIM);
	char temp[] = "conv"; 
	name = temp;
	for (int k = 0; k < 10; ++k) {
		cl_mem cl_in;
		cl_mem cl_weights;
		cl_mem cl_out;
		CNN_STOPWATCH(name) {
			// reset out
			for (size_t i = 0; i < OUT_DIM; i++) {
				out[i] = (cl_float) 0; 
			}
			cl_in = clCreateBuffer(context,
					CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
					IN_DIM * sizeof(cl_float), in, &errNum);
			cl_weights = clCreateBuffer(context,
					CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
					WEIGHTS_DIM * sizeof(cl_float), weights, &errNum);
			cl_out = clCreateBuffer(context,
					CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
					OUT_DIM * sizeof(cl_float), out, &errNum);

			// run function
			cnn(cl_in, cl_weights, cl_out, IN_DIM, &program, name);
			// get memory from gpu
			//printf("sizeof(out): %d OUT_DIM * sizeof(cl_float): %d", sizeof(out), OUT_DIM * sizeof(cl_float));
			errNum = clEnqueueReadBuffer(commandQueue,
					cl_out,
					CL_TRUE,
					0,
					sizeof(out),
					out,
					0,
					NULL,
					NULL);
			if (errNum != 0) {
				printf("Error Status %d in clEnqueueReadBuffer\n", errNum);
				exit(-1);
			}
		}

		//printf("\n\nOUT");
		//printvalues(out, OUT_DIM);
		clReleaseMemObject(cl_in);
		clReleaseMemObject(cl_out);
	}
	printf("GPU __IN_DIM: %4.lu TIME: %7.d TIME_PER_ELEMENT: %5.lu\n", IN_DIM, min(name, 0), min(name, 0) / IN_DIM);
	//printf("\n\n");

	char temp2[] = "float4conv"; 
	name = temp2;
	for (int k = 0; k < 10; ++k) {
		cl_mem cl_in;
		cl_mem cl_weights;
		cl_mem cl_out;
		char name[] = "float4conv";
		CNN_STOPWATCH(name) {
			// reset out
			for (size_t i = 0; i < OUT_DIM4; i++) {
				out4[i] = (cl_float4) {static_cast<cl_float>(0), static_cast<cl_float>(0), static_cast<cl_float>(0), static_cast<cl_float>(0)};
			}
			cl_in = clCreateBuffer(context,
					CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
					IN_DIM * sizeof(cl_float), in, &errNum);
			cl_weights = clCreateBuffer(context,
					CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
					WEIGHTS_DIM4 * sizeof(cl_float4), weights4, &errNum);
			cl_out = clCreateBuffer(context,
					CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
					OUT_DIM4 * sizeof(cl_float4), out4, &errNum);

			// run function
			cnn(cl_in, cl_weights, cl_out, IN_DIM, &program, name);
			// get memory from gpu
			//printf("sizeof(out4): %d OUT_DIM4 * sizeof(cl_float4): %d", sizeof(out4), OUT_DIM4 * sizeof(cl_float4));
			errNum = clEnqueueReadBuffer(commandQueue,
					cl_out,
					CL_TRUE,
					0,
					sizeof(out4),
					out4,
					0,
					NULL,
					NULL);
			if (errNum != 0) {
				printf("Error Status %d in clEnqueueReadBuffer\n", errNum);
				exit(-1);
			}
		}
		//printf("\n\nFLOAT4OUT");
		//printvalues(out4, OUT_DIM4);
		clReleaseMemObject(cl_in);
		clReleaseMemObject(cl_out);
	}
	printf("GPUFloat4 __IN_DIM: %4.lu TIME: %7.d TIME_PER_ELEMENT: %5.lu\n", IN_DIM, min(name, 0), min(name, 0) / IN_DIM);

	//printf("F4IN_DIM: %4.lu TIME: %7.d TIME_PER_ELEMENT: %5.lu ", IN_DIM, min(IN_DIM, 1), min(IN_DIM, 1) / IN_DIM);

	//printf("time: %d nano seconds\nsize: %d\naverage: %d nano seconds/run\n", total_elapsed, IN_DIM - 1, total_elapsed/IN_DIM);
	return 0;
}

int min(char* s1, int f4) {
	int min {INT32_MAX};
	int avg = 0;
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

