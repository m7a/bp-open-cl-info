#define N "\n"

const char* COPYRIGHT =

      /*##################################################################*/
      /*#*/                                                            /*#*/
      /*#*/ "Ma_Sys.ma OpenCL info and testing program 1.0.0, "        /*#*/
      /*#*/ "Copyright (c) 2016 Ma_Sys.ma."                            /*#*/ N
      /*#*/ "For further info send an e-mail to Ma_Sys.ma@web.de."     /*#*/ N
      /*#*/                                                            /*#*/
      /*##################################################################*/ ;

const char* GPL[] = {
      "This program is free software: you can redistribute it and/or modify"  N
      "it under the terms of the GNU General Public License as published by"  N
      "the Free Software Foundation, either version 3 of the License, or"     N
      "(at your option) any later version."                                   N
                                                                              N
      "This program is distributed in the hope that it will be useful,"       N
      "but WITHOUT ANY WARRANTY; without even the implied warranty of"        N
      "MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the"         N
      "GNU General Public License for more details."                          ,
                                                                              N
      "You should have received a copy of the GNU General Public License"     N
      "along with this program.  If not, see <http://www.gnu.org/licenses/>." N
};
 
/*
 * Compilation
 * $ gcc -fopenmp -lOpenCL -std=c89 -Wall -o ma_open_cl_info ma_open_cl_info.c
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

/*
 * http://dhruba.name/2012/08/14/opencl-cookbook-listing-all-devices-and-
 *   						their-critical-attributes/
 * https://developer.apple.com/library/mac/samplecode/
 * 				OpenCL_Hello_World_Example/Listings/hello_c.html
 */

#ifdef __APPLE__
#	include <OpenCL/opencl.h>
#else
#	include <CL/cl.h>
#endif

#define MULTIPLY_REPEAT 64 /* need to be MULTIPLY_REPEAT = 2^X, X in N */
#define EPSILON 0.00000001f
#define RAND_UPPER_BOUND 0.09f
#define DEFAULT_PROBLEM_SIZE 2048lu

#define INFOBUFSIZ 512

struct platform_info {
	char require_gpu;

	/** OpenGL platform */
	cl_uint num_platforms;
	cl_platform_id* platforms;
	cl_uint* num_devices;
	cl_device_id** devices;

	/** Tests */
	char skip_cpu;
	unsigned long problem_size;
	/* unidimensional notation for 2D matrices (more Open-CL like) */
	float* example_a;
	float* example_b;
	float* result;
};

static void display_help(char* appname);
static int run(unsigned long problem_size, char skip_cpu, char require_gpu);
static void ma_opencl_error(cl_int id, int line);
static char* get_error_string(cl_int code);
static void for_each_platform(struct platform_info* p,
				void (*func)(struct platform_info*, int));
static void get_and_print_platform_info(struct platform_info* p, int pid);
static void get_and_print_device_info(int j, cl_device_id d);
static void print_info_str(cl_device_id d, char* buf, int property,
							char* property_display);
static void print_info_siz(cl_device_id d, int property,
							char* property_display);
static void print_info_int(cl_device_id d, int property,
							char* property_display);
static void initialize_tests(struct platform_info* p);
static void matmul_sub(struct platform_info* p, long unsigned i,
							float* ca, float* cb);
static float my_randf();
static void delta_t(time_t t0, time_t t1);
static void calculate_on_platform(struct platform_info* p, int pid);
static void calculate_on_device(struct platform_info* p, int did,
								cl_device_id d);
static void report_program_build_failure(cl_program prg, cl_device_id d);
static void await_event(cl_event* event);
static void compare_check_equality(unsigned long n, float* expected,
								float* got);
static void free_tests(struct platform_info* p);

/*
	TODO PROBLEM: DOES NOT WORK WITH LARGE MATRICES BECAUSE THE KERNEL'S
	EXECUTION TIME IS SO TIGHTLY LIMITED IT IS EASILY EXCEEDED IF THE INNER
	LOOP GOES OVER MORE THAN ABOUT 100 MiB...
*/

const char* MATRIX_MULTIPLICATION_KERENEL_SRC =
	"__kernel void matmul(unsigned long problem_size,"                    \
	"		__global const float* a, __global const float* b, "   \
	"		__global float* result)"                              \
	"{"                                                                   \
	"	unsigned long j = get_global_id(0); /* get_g..._size(idx) */" \
	"	unsigned long i = get_global_id(1);"                          \
	"	unsigned long k;"                                             \
	"	float sum = 0.0f;"                                            \
	"	for(k = 0; k < problem_size; k++)"                            \
	"		sum += a[i * problem_size + k] * "                    \
	"					b[k * problem_size + j];"     \
	"	result[i * problem_size + j] = sum;"                          \
	"}";

int main(int argc, char** argv)
{
	int ca;
	unsigned long problem_size = DEFAULT_PROBLEM_SIZE;
	char skip_cpu = 0;
	char require_gpu = 0;

	puts(COPYRIGHT);
	puts(GPL[0]);
	puts(GPL[1]);

	for(ca = 1; ca < argc; ca++) {
		switch(argv[ca][1]) {
		case 'h':
		case '-':
			display_help(argv[0]);
			return EXIT_SUCCESS;
		case 'p':
			if(ca >= argc - 1) {
				display_help(argv[0]);
				return EXIT_FAILURE;
			} else {
				problem_size = atol(argv[++ca]);
				printf("Info: Changed problem size to %lu\n",
								problem_size);
			}
			break;
		case 'n':
			printf("Info: Disabled CPU result calculation\n");
			skip_cpu = 1;
			break;
		case 'g':
			printf("Info: OpenCL limited to GPU\n");
			require_gpu = 1;
		case 0:
			printf("Warning: Ignored single-letter option: %c\n",
								argv[ca][0]);
			break;
		default:
			printf("Warning: Ignored unknown option: %c\n",
								*argv[ca]);
		}
	}
	return run(problem_size, skip_cpu, require_gpu);
}

static void display_help(char* appname)
{
	printf(
		"USAGE %s [-h|--help] [-n] [-g] [-p <N>]\n\n"
		"-h  Displays this help.\n"
		"-n  Disables result verification (no CPU precalculation)\n"
		"-g  Requires OpenCL to be run on GPUs only. "
							"(Does not imply -n)\n"
		"-p  Configure problem size to be N, default %lu\n",
		appname, DEFAULT_PROBLEM_SIZE
	);
	puts(
"\nInformation on how this works\n\n"
"This program multiplies two random float/single-matrices A and B.\n"
"A and B are nxn-matrices where n is the ``problem size''.\n"
"In order to caluclate the result, the OpenCL device needs to provide three\n"
"times the memory necessary to hold one matrix (A, B and a result = 3 Matrices)"
	);
	printf(
"To be able to compute a comparison-result on CPU we not simply calculate\n"
"A x B, but instead (A x B)^%d. This way we can utilize\n"
"exponentiation by squaring on the CPU. In order to run longer on the GPU\n"
"(for more reliable measures), we do _not_ use exponentiation by squaring on\n"
"the OpenCL device.\n", MULTIPLY_REPEAT
	);
}

#define OCCALL(I, E) if((ret = (I)) != CL_SUCCESS) { \
					ma_opencl_error(ret, __LINE__); E; }

static int run(unsigned long problem_size, char skip_cpu, char require_gpu)
{
	cl_int ret;
	struct platform_info p;
	p.problem_size = problem_size;
	p.skip_cpu = skip_cpu;
	p.require_gpu = require_gpu;

	OCCALL(clGetPlatformIDs(0, NULL, &p.num_platforms),
							return EXIT_FAILURE);

	p.platforms = malloc(p.num_platforms * sizeof(cl_platform_id));
	OCCALL(clGetPlatformIDs(p.num_platforms, p.platforms, NULL),
					free(p.platforms); return EXIT_FAILURE);

	p.num_devices = malloc(p.num_platforms * sizeof(cl_uint));
	p.devices = malloc(p.num_platforms * sizeof(cl_device_id*));
	puts("System information");
	for_each_platform(&p, get_and_print_platform_info);
	puts("\nTests");
	initialize_tests(&p);
	for_each_platform(&p, calculate_on_platform);
	free_tests(&p);
	free(p.devices);
	free(p.num_devices);

	free(p.platforms);
	return EXIT_SUCCESS;
}

static void ma_opencl_error(cl_int id, int line)
{
	printf("Failed to run OpenCL statement, line %d, error code %d / %s.\n",
						line, id, get_error_string(id));
}

/* -> http://stackoverflow.com/questions/24326432/convenient-way-to-show-opencl-
								error-codes */
static char* get_error_string(cl_int code)
{
	switch(code) {
	case 0:     return "CL_SUCCESS";
	case -1:    return "CL_DEVICE_NOT_FOUND";
	case -2:    return "CL_DEVICE_NOT_AVAILABLE";
	case -3:    return "CL_COMPILER_NOT_AVAILABLE";
	case -4:    return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
	case -5:    return "CL_OUT_OF_RESOURCES";
	case -6:    return "CL_OUT_OF_HOST_MEMORY";
	case -7:    return "CL_PROFILING_INFO_NOT_AVAILABLE";
	case -8:    return "CL_MEM_COPY_OVERLAP";
	case -9:    return "CL_IMAGE_FORMAT_MISMATCH";
	case -10:   return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
	case -11:   return "CL_BUILD_PROGRAM_FAILURE";
	case -12:   return "CL_MAP_FAILURE";
	case -13:   return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
	case -14:   return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
	case -15:   return "CL_COMPILE_PROGRAM_FAILURE";
	case -16:   return "CL_LINKER_NOT_AVAILABLE";
	case -17:   return "CL_LINK_PROGRAM_FAILURE";
	case -18:   return "CL_DEVICE_PARTITION_FAILED";
	case -19:   return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

	case -30:   return "CL_INVALID_VALUE";
	case -31:   return "CL_INVALID_DEVICE_TYPE";
	case -32:   return "CL_INVALID_PLATFORM";
	case -33:   return "CL_INVALID_DEVICE";
	case -34:   return "CL_INVALID_CONTEXT";
	case -35:   return "CL_INVALID_QUEUE_PROPERTIES";
	case -36:   return "CL_INVALID_COMMAND_QUEUE";
	case -37:   return "CL_INVALID_HOST_PTR";
	case -38:   return "CL_INVALID_MEM_OBJECT";
	case -39:   return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
	case -40:   return "CL_INVALID_IMAGE_SIZE";
	case -41:   return "CL_INVALID_SAMPLER";
	case -42:   return "CL_INVALID_BINARY";
	case -43:   return "CL_INVALID_BUILD_OPTIONS";
	case -44:   return "CL_INVALID_PROGRAM";
	case -45:   return "CL_INVALID_PROGRAM_EXECUTABLE";
	case -46:   return "CL_INVALID_KERNEL_NAME";
	case -47:   return "CL_INVALID_KERNEL_DEFINITION";
	case -48:   return "CL_INVALID_KERNEL";
	case -49:   return "CL_INVALID_ARG_INDEX";
	case -50:   return "CL_INVALID_ARG_VALUE";
	case -51:   return "CL_INVALID_ARG_SIZE";
	case -52:   return "CL_INVALID_KERNEL_ARGS";
	case -53:   return "CL_INVALID_WORK_DIMENSION";
	case -54:   return "CL_INVALID_WORK_GROUP_SIZE";
	case -55:   return "CL_INVALID_WORK_ITEM_SIZE";
	case -56:   return "CL_INVALID_GLOBAL_OFFSET";
	case -57:   return "CL_INVALID_EVENT_WAIT_LIST";
	case -58:   return "CL_INVALID_EVENT";
	case -59:   return "CL_INVALID_OPERATION";
	case -60:   return "CL_INVALID_GL_OBJECT";
	case -61:   return "CL_INVALID_BUFFER_SIZE";
	case -62:   return "CL_INVALID_MIP_LEVEL";
	case -63:   return "CL_INVALID_GLOBAL_WORK_SIZE";
	case -64:   return "CL_INVALID_PROPERTY";
	case -65:   return "CL_INVALID_IMAGE_DESCRIPTOR";
	case -66:   return "CL_INVALID_COMPILER_OPTIONS";
	case -67:   return "CL_INVALID_LINKER_OPTIONS";
	case -68:   return "CL_INVALID_DEVICE_PARTITION_COUNT";

	case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
	case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
	case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
	case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
	case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
	case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
	default:    return "Unknown OpenCL error";
	}
}

static void for_each_platform(struct platform_info* p,
				void (*func)(struct platform_info*, int))
{
	int i;
	for(i = 0; i < p->num_platforms; i++)
		func(p, i);
}

static void get_and_print_platform_info(struct platform_info* p, int pid)
{
	cl_int ret;
	int j;
	long unsigned device_types = p->require_gpu? CL_DEVICE_TYPE_GPU:
							CL_DEVICE_TYPE_ALL;
	cl_platform_id cid = p->platforms[pid];

	printf("Platform %d: ", pid);
	OCCALL(clGetDeviceIDs(cid, device_types, 0, NULL, p->num_devices + pid),
					p->num_devices[pid] = 0; return);
	printf("%d devices.\n", p->num_devices[pid]);
	if(p->num_devices[pid] == 0)
		return;
	p->devices[pid] = malloc(p->num_devices[pid] * sizeof(cl_device_id));
	OCCALL(clGetDeviceIDs(cid, device_types, p->num_devices[pid],
					p->devices[pid], NULL), exit(64));

	for(j = 0; j < p->num_devices[pid]; j++)
		get_and_print_device_info(j, p->devices[pid][j]);
}

static void get_and_print_device_info(int j, cl_device_id d)
{
	char info[INFOBUFSIZ];
	printf("  Device %d:\n", j);
	print_info_str(d, info, CL_DEVICE_VENDOR,            "Vendor:");
	print_info_str(d, info, CL_DEVICE_NAME,              "Name:");
	print_info_int(d,       CL_DEVICE_MAX_COMPUTE_UNITS, "Compute Units:");
	print_info_siz(d,       CL_DEVICE_GLOBAL_MEM_SIZE,   "Global Memory:");
	print_info_siz(d,       CL_DEVICE_LOCAL_MEM_SIZE,    "Local Memory:");
}

static void print_info_str(cl_device_id d, char* buf, int property,
							char* property_display)
{
	cl_int ret;
	printf("    %-14s ", property_display);
	OCCALL(clGetDeviceInfo(d, property, INFOBUFSIZ, buf, NULL), return);
	puts(buf);
}

static void print_info_siz(cl_device_id d, int property, char* property_display)
{
	cl_int ret;
	unsigned long val;
	printf("    %-14s ", property_display);
	OCCALL(clGetDeviceInfo(d, property, sizeof(cl_ulong), &val, NULL),
									return);

	if(val > 1048576)
		printf("%lu MiB\n", val / 1024 / 1024);
	else if(val > 1024)
		printf("%lu KiB\n", val / 1024);
	else
		printf("%lu B\n", val);
}

static void print_info_int(cl_device_id d, int property, char* property_display)
{
	cl_int ret;
	cl_uint val;
	printf("    %-14s ", property_display);
	OCCALL(clGetDeviceInfo(d, property, sizeof(cl_uint), &val, NULL),
									return);
	printf("%u\n", val);
}

static void initialize_tests(struct platform_info* p)
{
	long unsigned asz;
	long unsigned i;
	int crep;
	float* r2;
	float* ca;
	float* cb;
	time_t t[4];

	asz = p->problem_size * p->problem_size;
	printf("Initializing Tests (asz=%lu MiB) ... ", asz * sizeof(float) /
								1024 / 1024);
	fflush(stdout);

	time(t);
	p->example_a = malloc(asz * sizeof(float));
	p->example_b = malloc(asz * sizeof(float));
	p->result = p->skip_cpu? NULL: malloc(asz * sizeof(float));

	time(t + 1);
	for(i = 0; i < asz; i++) {
		p->example_a[i] = my_randf();
		p->example_b[i] = my_randf();
	}

	time(t + 2);
	if(!p->skip_cpu) {
		/* TODO STRANGE BUG / IT SEEMS THIS IS NOT REALLY FASTER THAN THE OMP VARIANT ON THE VERY SAME DEVICE (CPU) ... SOMEHOW MESSED UP EFFICIENT MATRIX MULTIPLICATION? */
		r2 = malloc(asz * sizeof(float));
		ca = p->example_a;
		cb = p->example_b;
		for(crep = MULTIPLY_REPEAT; crep > 0; crep >>= 1) {
			#pragma omp parallel for
			for(i = 0; i < p->problem_size; i++)
				matmul_sub(p, i, ca, cb);

			ca = p->result;
			cb = p->result;
			p->result = r2;
			r2 = ca;
		}
		free(p->result);
		p->result = r2;
	}

	time(t + 3);

	printf("talloc=");
	delta_t(t[0], t[1]);
	printf(" trnd=");
	delta_t(t[1], t[2]);
	printf(" tcalc=");
	delta_t(t[2], t[3]);
	printf(" tS=");
	delta_t(t[0], t[3]);
	putchar('\n');
}

static void matmul_sub(struct platform_info* p, long unsigned i,
							float* ca, float* cb)
{
	long unsigned j;
	long unsigned k;
	float sum;
	for(j = 0; j < p->problem_size; j++) {
		sum = 0.0f;
		for(k = 0; k < p->problem_size; k++)
			sum += ca[i * p->problem_size + k] *
						cb[k * p->problem_size + j];
		p->result[i * p->problem_size + j] = sum;
	}
}

static float my_randf()
{
	float val = (float)((double)rand()/(double)(RAND_MAX/RAND_UPPER_BOUND));
	return (rand() >= (RAND_MAX/2))? val: -val;
}

static void delta_t(time_t t0, time_t t1)
{
	printf("%0.1f", difftime(t1, t0));
}

static void calculate_on_platform(struct platform_info* p, int pid)
{
	int i;
	if(p->num_devices[pid] == 0)
		return;

	printf("Platform %d\n", pid);
	for(i = 0; i < p->num_devices[pid]; i++)
		calculate_on_device(p, i, p->devices[pid][i]);

	free(p->devices[pid]);
}

#define OCCALLB(I, M, E) if(!(I)) { printf("    %s: ", M); \
					ma_opencl_error(ret, __LINE__); E; }

static void calculate_on_device(struct platform_info* p, int did,
								cl_device_id d)
{
	cl_int ret = CL_SUCCESS;
	cl_context context;
	cl_command_queue queue;
	cl_program prg;
	cl_kernel kernel;
	cl_event event;

	cl_mem a;
	cl_mem b;
	cl_mem result;

	cl_mem* ca;
	cl_mem* cb;
	cl_mem* cr;
	cl_mem* bak = NULL;

	float* result_ocl;
	int crep;
	size_t wdsz[2];
	unsigned long n = p->problem_size * p->problem_size;
	unsigned long memsz = n * sizeof(float);
	time_t t[6];

	printf("  Device %d\n", did);
	result_ocl = malloc(memsz);
	wdsz[0] = p->problem_size;
	wdsz[1] = p->problem_size;

	time(t);
	OCCALLB(context = clCreateContext(NULL, 1, &d, NULL, NULL, &ret),
			"Compute Context could not be created", goto r_r);
	OCCALLB(queue = clCreateCommandQueue(context, d, 0, &ret),
			"Command Queue could not be created", goto r_ctx);
	OCCALLB(prg = clCreateProgramWithSource(context, 1,
			&MATRIX_MULTIPLICATION_KERENEL_SRC, NULL, &ret),
			"Program could not be created", goto r_q);
	OCCALL(clBuildProgram(prg, 0, NULL, NULL, NULL, NULL),
			report_program_build_failure(prg, d); goto r_prg);
	OCCALLB(kernel = clCreateKernel(prg, "matmul", &ret),
			"Failed to create kernel", goto r_prg);

	time(t + 1);
	OCCALLB(a = clCreateBuffer(context,
				CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
				memsz, p->example_a, &ret),
				"Failed to allocate buffer `a`", goto r_prg);
	OCCALLB(b = clCreateBuffer(context,
				CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
				memsz, p->example_b, &ret),
				"Failed to allocate buffer `b`", goto r_ba);
	OCCALLB(result = clCreateBuffer(context, CL_MEM_READ_WRITE, memsz, NULL,
				&ret),
				"Failed to allocate result buffer", goto r_bb);

	/* CALCULATION */
	time(t + 2);
	ca = &a;
	cb = &b;
	cr = &result;
	OCCALL(clSetKernelArg(kernel, 0, sizeof(unsigned long),
						&p->problem_size), goto r_br);
	for(crep = MULTIPLY_REPEAT; crep > 0; crep--) {
		if(crep == MULTIPLY_REPEAT - 1) {
			/* cr:R, ca:A, cb:B, bak:X -> cr:A, ca:R, cb:R, bak:B */
			bak = cb; /* bak:X -> bak:B */
			cb  = cr; /* cb:B  -> cb:R  */
			cr  = ca; /* cr:R  -> cr:A  */
			ca  = cb; /* ca:A  -> ca:R  */
		} else if(crep != MULTIPLY_REPEAT) {
			/* cr:A, ca:R, cb:R, bak:B -> cr:B, ca:A, cb:R, bak:A */
			/* cr:B, ca:A, cb:R, bak:A -> cr:A, ca:B, cb:R, bak:B */
			/* cr:A, ca:B, cb:R, bak:B -> cr:B, ca:A, cb:R, bak:A */
			ca = cr;  /* ca:R  -> ca:A  */
			cr = bak; /* cr:A  -> cr:B  */
			bak = ca; /* bak:B -> bak:A */
		}
		OCCALL(clSetKernelArg(kernel, 1, sizeof(cl_mem), ca),
								goto r_br);
		OCCALL(clSetKernelArg(kernel, 2, sizeof(cl_mem), cb),
								goto r_br);
		OCCALL(clSetKernelArg(kernel, 3, sizeof(cl_mem), cr),
								goto r_br);

		/* INVOC */
		OCCALL(clEnqueueNDRangeKernel(queue, kernel, 2, NULL, wdsz,
					NULL, 0, NULL, &event), goto r_br);
		await_event(&event);
	}

	time(t + 3);
	OCCALL(clEnqueueReadBuffer(queue, *cr, CL_TRUE, 0, memsz, result_ocl, 0,
						NULL, &event), goto r_br);
	await_event(&event);

	time(t + 4);
	if(!p->skip_cpu)
		compare_check_equality(n, p->result, result_ocl);

	time(t + 5);
	printf("    tinit=");
	delta_t(t[0], t[1]);
	printf(", tmem=");
	delta_t(t[1], t[2]);
	printf(", tcalc=");
	delta_t(t[2], t[3]);
	printf(", tmem2=");
	delta_t(t[3], t[4]);
	printf(", tcmp=");
	delta_t(t[4], t[5]);
	printf(", tS/OCL=");
	delta_t(t[0], t[4]);
	printf(", tS=");
	delta_t(t[0], t[5]);
	putchar('\n');

r_br:   OCCALL(clReleaseMemObject(result), ;);
r_bb:   OCCALL(clReleaseMemObject(b), ;);
r_ba:   OCCALL(clReleaseMemObject(a), ;);
r_prg:  OCCALL(clReleaseProgram(prg), ;);
r_q:    OCCALL(clReleaseCommandQueue(queue), ;);
r_ctx:  OCCALL(clReleaseContext(context), ;);
r_r:    free(result_ocl);
}

static void report_program_build_failure(cl_program prg, cl_device_id d)
{
	cl_int ret;
        char buf[4096];
	printf("Failed to compile program: ");
	OCCALL(clGetProgramBuildInfo(prg, d, CL_PROGRAM_BUILD_LOG, sizeof(buf),
							buf, NULL), return);
        printf("%s\n", buf);
}

static void await_event(cl_event* event)
{
	cl_int ret;
	OCCALL(clWaitForEvents(1, event), ;);
	OCCALL(clReleaseEvent(*event), ;);
}

static void compare_check_equality(unsigned long n, float* expected, float* got)
{
	unsigned long i;
	printf("    Comparing results... ");
	for(i = 0; i < n; i++)
		if(abs(expected[i] - got[i]) > EPSILON)
			printf("Mismatch@%lu: delta=%f ", i,
							expected[i] - got[i]);
	printf("finished\n");
}

static void free_tests(struct platform_info* p)
{
	free(p->example_a);
	free(p->example_b);
	if(!p->skip_cpu)
		free(p->result);
}
