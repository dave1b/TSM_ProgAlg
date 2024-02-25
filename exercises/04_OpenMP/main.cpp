// C++ is one-pass compiler. If we want to use a function/procedure/method, then it has to be
// defined or at least declared before.
// Here are two function declartions. The implementations of these function are in different
// compilation units: imageprocessing.cpp and summation.cpp
// It's the task of the C++ linker to put all parts of a program together.
void summation();
void matrixRowSorting();

// main program
int main() {
	summation();
	matrixRowSorting();
}