#include <iostream>
#include <map>
#include <vector>
#include <chrono>
#include <string>
#include <CL/cl.h>
#include <cstdlib>
#include <cstdio>
#include <sys/time.h>

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



int main()
{
    const int size = 50;
    const int stepsize = 1;

    if (initOcl()) {
        fprintf(stderr,"Failed InitOcl\n");
        exit(-1);
    }
    //initclmemobjects();
    float *in;
    float *out;
    size_t IN_DIM;
    size_t OUT_DIM;
    int total_elapsed = 0;
    for (OUT_DIM = IN_DIM = 1; IN_DIM <= size; OUT_DIM = IN_DIM += stepsize) {
        for (int j = 0; j < 10; ++j) {
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
        printf("IN_DIM: %4.lu TIME: %7.d TIME_PER_ELEMENT: %5.lu\n", IN_DIM, min(IN_DIM), min(IN_DIM) / IN_DIM);
        total_elapsed += min(IN_DIM);
    }

    //printf("time: %d nano seconds\nsize: %d\naverage: %d nano seconds/run\n", total_elapsed, IN_DIM - 1, total_elapsed/IN_DIM);

    return 0;
}
int min(int index) {
    int min {INT32_MAX};
    for (size_t i = 0; i < stopwatch_timings[std::to_string(index).c_str()].size(); ++i) {
        min = min <= stopwatch_timings[std::to_string(index).c_str()][i] ? min : stopwatch_timings[std::to_string(index).c_str()][i];
    }
    return min;
}


