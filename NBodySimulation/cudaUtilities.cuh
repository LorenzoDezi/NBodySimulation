
#define CHECK(call) {  \
	cudaError_t cudaStatus = call; \
	if (cudaStatus != cudaSuccess) { \
		std::cout << "Error: " << __FILE__ << " at " << __LINE__ << std::endl; \
		std::cout << "Code: "<< cudaStatus << "; Reason: " << cudaGetErrorString(cudaStatus) << std::endl; \
		return 1; \
	} \
} \