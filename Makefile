CC = gcc
VALGRINDOPTS = valgrind --log-file="logfile" --leak-check=full --show-leak-kinds=all --track-origins=yes --dsymutil=yes --trace-children=yes -v

# for multithreading
# -fipa-pta -floop-nest-optimize -floop-parallelize-all -ftree-loop-distribution -ftree-parallelize-loops=4 -ftree-vectorize -funroll-loops -flto -ffat-lto-objects -fuse-linker-plugin -funsafe-math-optimizations -freciprocal-math 

.PHONY: all
all: test1 test2 test3 test4

test1:
	cd ./examples/;\
	$(CC) -O3 -march=native -Wall -std=c11 -o ./test1 ./mlp_reg.c -lm;

test1db:
	cd ./examples/;\
	$(CC) -march=native -Wall -std=c11 -g -o ./test1 ./mlp_reg.c -lm;

valtest1:
	cd ./examples/;\
	$(CC) -march=native -Wall -std=c11 -g -o ./test1 ./mlp_reg.c -lm;\
	$(VALGRINDOPTS) ./test1;

test2:
	cd ./examples/;\
	$(CC) -O3 -march=native -Wall -std=c11 -o ./test2 ./mlp_classification.c -lm;

test2db:
	cd ./examples/;\
	$(CC) -march=native -Wall -std=c11 -g -o ./test2 ./mlp_classification.c -lm;
	
valtest2:
	cd ./examples/;\
	$(CC) -march=native -Wall -std=c11 -g -o ./test1 ./mlp_classification.c -lm;\
	$(VALGRINDOPTS) ./test2;

.PHONY: test1
test1:
	cd ./examples/;\
	$(CC) -O3 -march=native -Wall -std=c11 -o ./test1 ./mlp_reg.c -lm;
	
.PHONY: test2
test2:
	cd ./examples/;\
	$(CC) -O3 -march=native -Wall -std=c11 -o ./test2 ./mlp_classification.c -lm;
