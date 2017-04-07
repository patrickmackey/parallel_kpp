CC = nvcc
CFLAGS =
LDFLAGS =
LD = nvcc
EXE = cudatest
OBJS = tools.o kkz.o kpp.o kmeans.o main.o

all: $(EXE)

$(EXE): $(OBJS)
	$(LD) $(LDFLAGS) -o $@ $^

main.o: main.cu
	$(CC) $(CFLAGS) -c $< -o $@

kmeans.o: kmeans.cu
	$(CC) $(CFLAGS) -c $< -o $@

kkz.o: kkz.cu
	$(CC) $(CFLAGS) -c $< -o $@

kpp.o: kpp.cu
	$(CC) $(CFLAGS) -c $< -o $@

tools.o: tools.cu
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(EXE) $(OBJS)

