#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wwritable-strings"
#define CNN_TEST
#ifdef CNN_TEST
#include <iostream>
#include <map>
#include <vector>
#include <chrono>
#include <string>

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
#else
#define CNN_STOPWATCH(name) STOPWATCH(name)
#endif
#define LAYER_STOPWATCH
#ifdef LAYER_STOPWATCH
#define INTERNAL_CNN_STOPWATCH(name) CNN_STOPWATCH(name)
#else
#define INTERNAL_CNN_STOPWATCH(name) // CNN_STOPWATCH(name)
#endif

// includes
#include <CL/cl.h>
#include <stdlib.h>

cl_int errNum;
cl_context context;
cl_command_queue commandQueue;
cl_program program;
/*
// buffers
float separable_conv2d_6_internal_1_W_[] = {-1.0299174785614014f,-0.7187300324440002f,0.41224345564842224f,0.42398306727409363f,-0.5002785921096802f,-2.3287012577056885f,-0.8622229695320129f,-1.4436947107315063f,-0.14598743617534637f,-0.663096010684967f,0.26683396100997925f,-0.34726014733314514f,-1.4952480792999268f,-0.8092780113220215f,-0.2202264964580536f,-0.1287665218114853f,0.13643066585063934f,-3.5613629817962646f,-1.9674489498138428f,-1.7091423273086548f,-0.5929750800132751f,-1.4317723512649536f,0.42840293049812317f,-0.4904371500015259f,-0.38027000427246094f,-0.6622756123542786f,-0.03256722167134285f,-0.21384072303771973f,0.1911373883485794f,-2.522857666015625f,-1.6847249269485474f,-0.6022032499313354f,-0.3444325625896454f,-1.6270148754119873f,0.53294837474823f,-0.5188931226730347f,0.8265866637229919f,-2.00363826751709f,-0.15832051634788513f,0.05767636373639107f,0.5321109294891357f,-0.32593676447868347f,2.436944007873535f,-2.369814872741699f,2.4012961387634277f,-0.08473268896341324f,-0.03152402117848396f,-0.04355483874678612f,0.07742287218570709f,-1.8716108798980713f,0.1381376087665558f,0.00392795167863369f,-0.07512766122817993f,-1.4633091688156128f,0.3343474566936493f,-2.597665309906006f,2.09517765045166f,-1.602073311805725f,-1.200972557067871f,1.3932440280914307f,-0.4011599123477936f,-0.5751668214797974f,-0.30258965492248535f,-0.27163636684417725f,-0.3377749025821686f,-1.0814716815948486f,-0.9921945333480835f,-0.9431122541427612f,1.2839919328689575f,-0.7706644535064697f,-7.847440429031849e-06f,-0.010538669303059578f,2.164921998977661f,-3.2143146991729736f,0.3554742932319641f,0.34240758419036865f,-0.19093061983585358f,-0.7241868376731873f,1.9666281938552856f,1.203127384185791f,3.3089993000030518f,1.7075246572494507f,0.431784063577652f,-0.42256832122802734f,2.262066125869751f,-4.347748756408691f,-0.25184333324432373f,-0.3268728256225586f,-0.19181500375270844f,-0.6494271159172058f,2.6669399738311768f,1.1107572317123413f,3.9626684188842773f,2.0069737434387207f,-0.3475162088871002f,0.3949359059333801f,1.5718762874603271f,-1.8905502557754517f,0.14919805526733398f,0.11933018267154694f,0.2346019446849823f,-1.0056077241897583f,-0.10440247505903244f,1.2922465801239014f,1.8520045280456543f,-0.02102428674697876f,0.27546557784080505f,-0.3743652403354645f};
cl_mem separable_conv2d_6_internal_1_W;
void initclmemobjects()
{
separable_conv2d_6_internal_1_W = clCreateBuffer(context,
  CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR,
  108 * sizeof(float), separable_conv2d_6_internal_1_W_, &errNum);
}

//cl_buffers
  cl_mem x0;
      float * x0_result;
  cl_mem x1;
*/
//initOcl()

int min(int index);

const int COUNT = 9;
int initOcl()
{
const char * kernelSourceString[COUNT] = {
"  __kernel void testkernel(\n",
"    __global const float *in,\n",
"    __global float *out)\n",
"  {\n",
"      int index = get_global_id(0); //H_OUT\n",
"      float temp = in[index];\n",
"      temp *= 2.0f;\n",
"      out[index] = temp;\n",
"  }\n"
};

  cl_uint dev_cnt = 0;
  clGetPlatformIDs(0, 0, &dev_cnt);

  cl_platform_id platform_ids[100];
  clGetPlatformIDs(dev_cnt, platform_ids, NULL);

  cl_device_id device_id;
  errNum = clGetDeviceIDs(platform_ids[0], CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
  if (errNum != CL_SUCCESS)
  {
    printf("Error: Failed to create a device group!\n");
    return -1;
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

  program = clCreateProgramWithSource(context, COUNT, (const char **) kernelSourceString, NULL, &errNum);

  if (!program)
  {
    printf("Error: Failed to create compute program!\n");
    return EXIT_FAILURE;
  }

  errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  if (errNum != CL_SUCCESS)
  {
    size_t len;
    char buffer[2048];
    printf("Error: Failed to build program executable!\n");
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
    printf("%s\n", buffer);
    exit(1);
  }
  /*
  x0_result = (float*) calloc(58443, sizeof(float));
  x0 = clCreateBuffer(context,
    CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR,
    58443 * sizeof(float), x0_result, &errNum);
  x1 = clCreateBuffer(context,
    CL_MEM_READ_WRITE,
    57600 * sizeof(float), NULL, &errNum);
    */

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

void cnn(cl_mem in_0, cl_mem out_0, size_t dim)
{
  {
    {
        cl_kernel layer;
        size_t localWorkSize[1] = {1};
        size_t globalWorkSize[1] = {dim};
        {

            char a[] = "testkernel";
            createKernel(&program, &layer, a);

            if (in_0 == NULL || out_0 == NULL) {
                printf("Given memObjects are null in testkernel.\n");
                exit(-1);
            }
            errNum = clSetKernelArg(layer, 0, sizeof(cl_mem), (void *) &in_0);
            errNum |= clSetKernelArg(layer, 1, sizeof(cl_mem), (void *) &out_0);

            if (errNum != CL_SUCCESS) {
                printf("Error setting Kernel Arguments in testkernel.\n");
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

#ifdef CNN_TEST

#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

int main()
{
    const int size = 300;
    const int stepsize = 1;



    if (initOcl()) {
        fprintf(stderr,"Failed InitOcl\n");
        exit(-1);
    }
    //initclmemobjects();
    float *in;
    float *out;
    int IN_DIM;
    int OUT_DIM;
    int total_elapsed = 0;
    for (OUT_DIM = IN_DIM = 1; IN_DIM <= size; OUT_DIM = IN_DIM += stepsize) {
        for (int i = 0; i < 30; ++i) {
            in = (float *) malloc(IN_DIM * sizeof(float));
            out = (float *) malloc(OUT_DIM * sizeof(float));


            for (size_t i = 0; i < IN_DIM; i++) {
                in[i] = static_cast<float>(i);
            }
            cl_mem cl_in;
            cl_mem cl_out;
            cl_in = clCreateBuffer(context,
                                   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   IN_DIM * sizeof(float), in, &errNum);
            cl_out = clCreateBuffer(context,
                                    CL_MEM_READ_WRITE,
                                    OUT_DIM * sizeof(float), NULL, &errNum);

            // reset out
            for (size_t i = 0; i < OUT_DIM; i++) {
                out[i] = 0.0f;
            }

            // run function
            CNN_STOPWATCH(std::to_string(IN_DIM).c_str()) {
                cnn(cl_in, cl_out, IN_DIM);
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
            free(in);
            free(out);
        }
        printf("IN_DIM:%4.d TIME:%7.d TIME_PER_ELEMENT:%5.d\n", IN_DIM, min(IN_DIM), min(IN_DIM) / IN_DIM);
        total_elapsed += min(IN_DIM);
    }

    //printf("time: %d nano seconds\nsize: %d\naverage: %d nano seconds/run\n", total_elapsed, IN_DIM - 1, total_elapsed/IN_DIM);

    return 0;
}
int min(int index) {
    int min {INT32_MAX};
    for (int i = 0; i < stopwatch_timings[std::to_string(index).c_str()].size(); ++i) {
        min = min <= stopwatch_timings[std::to_string(index).c_str()][i] ? min : stopwatch_timings[std::to_string(index).c_str()][i];
    }
    return min;
}

#endif

#pragma clang diagnostic pop