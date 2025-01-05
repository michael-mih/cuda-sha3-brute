#include <algorithm>
#include <iostream>
#include <cstring>
#include <fstream>
// for mmap:
#include <windows.h>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <fcntl.h>
#include <string>
#include <tuple>

std::tuple<size_t, size_t> readFile(char const* fname, std::string& charArray);

void hexInputToBytes(const char* input, unsigned char* buffer, size_t buffsize);