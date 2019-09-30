
all: batcher.so

clean:
	rm batcher.so

TF_INC = $(shell python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB = $(shell python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

batcher.so:
	g++ -std=c++11 -shared batcher.cc -o batcher.so -fPIC -I $(TF_INC) -O2 -D_GLIBCXX_USE_CXX11_ABI=0 -L$(TF_LIB) -ltensorflow_framework
