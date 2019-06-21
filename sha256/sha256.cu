// cd /home/hork/cuda-workspace/CudaSHA256/Debug/files
// time ~/Dropbox/FIIT/APS/Projekt/CpuSHA256/a.out -f ../file-list
// time ../CudaSHA256 -f ../file-list


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <cuda.h>
#include "sha256.cuh"
#include <dirent.h>
#include <ctype.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <errno.h>

/************************************************************************************************************************
 * 1、int socket(int family,int type,int protocol)
 * family:
 *     指定使用的协议簇:AF_INET（IPv4） AF_INET6（IPv6） AF_LOCAL（UNIX协议） AF_ROUTE（路由套接字） AF_KEY（秘钥套接字）
 * type:
 *     指定使用的套接字的类型:SOCK_STREAM（字节流套接字） SOCK_DGRAM
 * protocol:
 *     如果套接字类型不是原始套接字，那么这个参数就为0
 * 2、int bind(int sockfd, struct sockaddr *myaddr, int addrlen)
 * sockfd:
 *     socket函数返回的套接字描述符
 * myaddr:
 *     是指向本地IP地址的结构体指针
 * myaddrlen:
 *     结构长度
 *struct sockaddr{
 *     unsigned short sa_family; //通信协议类型族AF_xx
 *     char sa_data[14];  //14字节协议地址，包含该socket的IP地址和端口号
 * };
 *
 * struct sockaddr_in{
 *     short int sin_family; //通信协议类型族
 *     unsigned short int sin_port; //端口号
 *     struct in_addr sin_addr; //IP地址
 *     unsigned char si_zero[8];  //填充0以保持与sockaddr结构的长度相同
 *};
 * 3、int connect(int sockfd,const struct sockaddr *serv_addr,socklen_t addrlen)
 *     sockfd:
 *            socket函数返回套接字描述符
 *     serv_addr:
 *            服务器IP地址结构指针
 *     addrlen:
 *            结构体指针的长度
 * 4、int listen(int sockfd, int backlog)
 *     sockfd:
 *            socket函数绑定bind后套接字描述符
 *     backlog:
 *            设置可连接客户端的最大连接个数，当有多个客户端向服务器请求时，收到此值的影响。默认值20
 * 5、int accept(int sockfd,struct sockaddr *cliaddr,socklen_t *addrlen)
 *     sockfd:
 *            socket函数经过listen后套接字描述符
 *     cliaddr:
 *            客户端套接字接口地址结构
 *     addrlen:
 *            客户端地址结构长度
 * 6、int send(int sockfd, const void *msg,int len,int flags)
 * 7、int recv(int sockfd, void *buf,int len,unsigned int flags)
 *     sockfd:
 *            socket函数的套接字描述符
 * msg:
 *     发送数据的指针
 * buf:
 *     存放接收数据的缓冲区
 * len:
 * 数据的长度，把flags设置为0
 **************************************************************************************************************************/


char * trim(char *str){
    size_t len = 0;
    char *frontp = str;
    char *endp = NULL;

    if( str == NULL ) { return NULL; }
    if( str[0] == '\0' ) { return str; }

    len = strlen(str);
    endp = str + len;

    /* Move the front and back pointers to address the first non-whitespace
     * characters from each end.
     */
    while( isspace((unsigned char) *frontp) ) { ++frontp; }
    if( endp != frontp )
    {
        while( isspace((unsigned char) *(--endp)) && endp != frontp ) {}
    }

    if( str + len - 1 != endp )
            *(endp + 1) = '\0';
    else if( frontp != str &&  endp == frontp )
            *str = '\0';

    /* Shift the string so that it starts at str so that if it's dynamically
     * allocated, we can still free it on the returned pointer.  Note the reuse
     * of endp to mean the front of the string buffer now.
     */
    endp = str;
    if( frontp != str )
    {
            while( *frontp ) { *endp++ = *frontp++; }
            *endp = '\0';
    }

    return str;
}

__global__ void sha256_cuda(JOB ** jobs, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	// perform sha256 calculation here
	if (i < n){
		SHA256_CTX ctx;
		sha256_init(&ctx);
		sha256_update(&ctx, jobs[i]->data, jobs[i]->size);
		sha256_final(&ctx, jobs[i]->digest);
	}
}

void pre_sha256() {
	// compy symbols
        checkCudaErrors(cudaMemcpyToSymbol(dev_k, host_k, sizeof(host_k), 0, cudaMemcpyHostToDevice));
//        int host_k = 256;
//	checkCudaErrors(cudaMemcpyToSymbol("d_nx", &host_k, sizeof(host_k), 0, cudaMemcpyHostToDevice));
}


void runJobs(JOB ** jobs, int n){
	int blockSize = 4;
	int numBlocks = (n + blockSize - 1) / blockSize;
	sha256_cuda <<< numBlocks, blockSize >>> (jobs, n);
}


JOB * JOB_init(BYTE * data, long size, char * fname) {
	JOB * j;
	checkCudaErrors(cudaMallocManaged(&j, sizeof(JOB)));	//j = (JOB *)malloc(sizeof(JOB));
	checkCudaErrors(cudaMallocManaged(&(j->data), size));
	j->data = data;
	j->size = size;
	for (int i = 0; i < 64; i++)
	{
		j->digest[i] = 0xff;
	}
	strcpy(j->fname, fname);
	return j;
}


BYTE * get_file_data(char * fname, unsigned long * size) {
	FILE * f = 0;
	BYTE * buffer = 0;
	unsigned long fsize = 0;

	f = fopen(fname, "rb");
	if (!f){
		fprintf(stderr, "get_file_data Unable to open '%s'\n", fname);
		return 0;
	}
	fflush(f);

	if (fseek(f, 0, SEEK_END)){
		fprintf(stderr, "Unable to fseek %s\n", fname);
		return 0;
	}
	fflush(f);
	fsize = ftell(f);
	rewind(f);

	//buffer = (char *)malloc((fsize+1)*sizeof(char));
	checkCudaErrors(cudaMallocManaged(&buffer, (fsize+1)*sizeof(char)));
	fread(buffer, fsize, 1, f);
	fclose(f);
	*size = fsize;
	return buffer;
}

void print_usage(){
	printf("Usage: CudaSHA256 [OPTION] [FILE]...\n");
	printf("Calculate sha256 hash of given FILEs\n\n");
	printf("OPTIONS:\n");
	printf("\t-f FILE1 \tRead a list of files (separeted by \\n) from FILE1, output hash for each file\n");
	printf("\t-h       \tPrint this help\n");
	printf("\nIf no OPTIONS are supplied, then program reads the content of FILEs and outputs hash for each FILEs \n");
	printf("\nOutput format:\n");
	printf("Hash following by two spaces following by file name (same as sha256sum).\n");
	printf("\nNotes:\n");
	printf("Calculations are performed on GPU, each seperate file is hashed in its own thread\n");
}

int main(int argc, char **argv) {
    while (1)
    {
        int fd, new_fd, struct_len, numbytes, i;
        struct sockaddr_in server_addr;
        struct sockaddr_in client_addr;
        char buff[BUFSIZ];

        server_addr.sin_family = AF_INET;
        server_addr.sin_port = htons(8000);
        server_addr.sin_addr.s_addr = INADDR_ANY;
        bzero(&(server_addr.sin_zero), 8);
        struct_len = sizeof(struct sockaddr_in);

        fd = socket(AF_INET, SOCK_STREAM, 0);
        while (bind(fd, (struct sockaddr *)&server_addr, struct_len) == -1)
            ;
        printf("Bind Success!\n");
        while (listen(fd, 10) == -1)
            ;
        printf("Listening....\n");
        printf("Ready for Accept,Waitting...\n");
        new_fd = accept(fd, (struct sockaddr *)&client_addr , (socklen_t *)&struct_len);
        printf("Get the Client.\n");
        numbytes = send(new_fd, "Welcome to my server\n", 21, 0);
        while ((numbytes = recv(new_fd, buff, BUFSIZ, 0)) > 0)
        {
            buff[numbytes] = '\0';
            printf("%s\n", buff);
            if (send(new_fd, buff, numbytes, 0) < 0)
            {
                perror("write");
                return 1;
            }
        }
        BYTE * buff_job=(BYTE *)buff;
        unsigned long temp;
        JOB ** jobs;
        int n=0; //生成随机数种子
        checkCudaErrors(cudaMallocManaged(&jobs, sizeof(JOB *)));
        jobs[n++] = JOB_init(buff_job, n, buff);
        pre_sha256();
        runJobs(jobs, n);
        cudaDeviceSynchronize();
	print_jobs(jobs, n);
        return 0;
        cudaDeviceReset();
        close(new_fd);
        close(fd);
    }
	return 0;
}