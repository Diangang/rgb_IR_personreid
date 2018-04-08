TORCH_INSTALL_DIR = /home/arul/torch/install
 
all: libNormCrossMapCorrelation.so libCrossInputNeighborhood.so libNormCrossMapSSIM.so

libNormCrossMapCorrelation.so: NormCrossMapCorrelation.cu
	nvcc --std=c++11 --shared --compiler-options '-fPIC -Wall -O2' -I $(TORCH_INSTALL_DIR)/include -I $(TORCH_INSTALL_DIR)/include/TH -o libNormCrossMapCorrelation.so NormCrossMapCorrelation.cu

libCrossInputNeighborhood.so: CrossInputNeighborhood.cu
	nvcc --std=c++11 --shared --compiler-options '-fPIC -Wall -O2' -I $(TORCH_INSTALL_DIR)/include -I $(TORCH_INSTALL_DIR)/include/TH -o libCrossInputNeighborhood.so CrossInputNeighborhood.cu

libNormCrossMapSSIM.so: NormCrossMapSSIM.cu
	nvcc --std=c++11 --shared --compiler-options '-fPIC -Wall -O2' -I $(TORCH_INSTALL_DIR)/include -I $(TORCH_INSTALL_DIR)/include/TH -o libNormCrossMapSSIM.so NormCrossMapSSIM.cu
