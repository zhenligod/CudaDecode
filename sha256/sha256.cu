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

char *trim(char *str)
{
	size_t len = 0;
	char *frontp = str;
	char *endp = NULL;

	if (str == NULL)
	{
		return NULL;
	}
	if (str[0] == '\0')
	{
		return str;
	}

	len = strlen(str);
	endp = str + len;

	/* Move the front and back pointers to address the first non-whitespace
     * characters from each end.
     */
	while (isspace((unsigned char)*frontp))
	{
		++frontp;
	}
	if (endp != frontp)
	{
		while (isspace((unsigned char)*(--endp)) && endp != frontp)
		{
		}
	}

	if (str + len - 1 != endp)
		*(endp + 1) = '\0';
	else if (frontp != str && endp == frontp)
		*str = '\0';

	/* Shift the string so that it starts at str so that if it's dynamically
     * allocated, we can still free it on the returned pointer.  Note the reuse
     * of endp to mean the front of the string buffer now.
     */
	endp = str;
	if (frontp != str)
	{
		while (*frontp)
		{
			*endp++ = *frontp++;
		}
		*endp = '\0';
	}
	return str;
}

__global__ void sha256_cuda(JOB **jobs, int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	// perform sha256 calculation here
	if (i < n)
	{
		SHA256_CTX ctx;
		sha256_init(&ctx);
		sha256_update(&ctx, jobs[i]->data, jobs[i]->size);
		sha256_final(&ctx, jobs[i]->digest);
	}
}

void pre_sha256()
{
	// compy symbols
	checkCudaErrors(cudaMemcpyToSymbol(dev_k, host_k, sizeof(host_k), 0, cudaMemcpyHostToDevice));
}

void runJobs(JOB **jobs, int n)
{
	int blockSize = 4;
	int numBlocks = (n + blockSize - 1) / blockSize;
	sha256_cuda<<<numBlocks, blockSize>>>(jobs, n);
}

JOB *JOB_init(BYTE *data, long size, char *fname)
{
	JOB *j;
	checkCudaErrors(cudaMallocManaged(&j, sizeof(JOB))); //j = (JOB *)malloc(sizeof(JOB));
	checkCudaErrors(cudaMallocManaged(&(j->data), size));
	j->data = data;
	printf("data %x", data);
	j->size = size;
	for (int i = 0; i < 64; i++)
	{
		j->digest[i] = 0xff;
	}
	strcpy(j->fname, fname);
	return j;
}

BYTE *get_file_data(char *fname, unsigned long *size)
{
	FILE *f = 0;
	BYTE *buffer = 0;
	unsigned long fsize = 0;

	f = fopen(fname, "rb");
	if (!f)
	{
		fprintf(stderr, "get_file_data Unable to open '%s'\n", fname);
		return 0;
	}
	fflush(f);

	if (fseek(f, 0, SEEK_END))
	{
		fprintf(stderr, "Unable to fseek %s\n", fname);
		return 0;
	}
	fflush(f);
	fsize = ftell(f);
	rewind(f);

	checkCudaErrors(cudaMallocManaged(&buffer, (fsize + 1) * sizeof(char)));
	fread(buffer, fsize, 1, f);
	fclose(f);
	*size = fsize;
	return buffer;
}

int main(int argc, char **argv)
{
	while (1)
	{
		int fd, new_fd, struct_len, numbytes;
		struct sockaddr_in server_addr;
		struct sockaddr_in client_addr;
		BYTE buff[BUFSIZ];

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
		new_fd = accept(fd, (struct sockaddr *)&client_addr, (socklen_t *)&struct_len);
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
		int i = 0, n = 0;
		size_t len;
		unsigned long temp;
		char *a_file = 0, *line = 0;
		BYTE *buff_sha256 = (BYTE *)buff;
		char option, index;
		ssize_t read;
		JOB **jobs;
		a_file = "test_file";
		if (a_file)
		{
			FILE *f = 0;
			f = fopen(a_file, "r");
			if (!f)
			{
				fprintf(stderr, "Unable to open %s\n", a_file);
				return 0;
			}
			for (n = 0; getline(&line, &len, f) != -1; n++)
			{
			}
			checkCudaErrors(cudaMallocManaged(&jobs, n * sizeof(JOB *)));
			fseek(f, 0, SEEK_SET);
			n = 0;
			read = getline(&line, &len, f);
			while (read != -1)
			{
				//printf("%s\n", line);
				read = getline(&line, &len, f);
				line = trim(line);
				FILE *fp;
				fp = fopen(line, "w");
				fprintf(fp, "%s\n", buff);
				fclose(fp);
				buff_sha256 = get_file_data(line, &temp);
				printf("buff_sha256 %x\n", buff_sha256);
				jobs[n++] = JOB_init(buff_sha256, temp, line);
			}
			pre_sha256();
			runJobs(jobs, n);
		}

		cudaDeviceSynchronize();
		print_jobs(jobs, n);
		cudaDeviceReset();
		close(new_fd);
		close(fd);
	}
	return 0;
}