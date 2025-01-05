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
#include <tuple>

std::tuple<size_t, size_t> readFile(char const* fname, std::string& charArray)
{
    static const auto BUFFER_SIZE = 16 * 1024; //needs optimize
    HANDLE file = CreateFileA(fname, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (file == INVALID_HANDLE_VALUE)
    {
        std::cerr << "Error: Unable to open file: " << fname << "\n";
        return { 0, 0 };
    }
    char buf[BUFFER_SIZE + 1];
    DWORD bytesRead;
    size_t lines = 0;
    size_t size = 0;
    while (ReadFile(file, buf, BUFFER_SIZE, &bytesRead, NULL) && bytesRead > 0)
    {
        buf[bytesRead] = '\0'; 
        //while p can still find a newline
        char* lag = buf;
        for (char* p = buf; (p = strchr(p, '\n')) != nullptr; ++p){
            *p = '\0';

            charArray += lag;
            charArray += '\0';
            size += p - lag;
            lines++;
            lag = p+1;
        }

    }

    CloseHandle(file);
    return { lines, size };
}

void hexInputToBytes(const char* input, unsigned char* buffer, size_t buffsize) {
    for (size_t i = 0; i < buffsize; ++i)
    {
        char byteHex[3] = { input[2 * i], input[2 * i + 1], '\0' };  // Two hex chars per byte
        if (!std::isxdigit(byteHex[0]) || !std::isxdigit(byteHex[1]))
            throw std::invalid_argument("Invalid hexadecimal character found.");
        buffer[i] = static_cast<unsigned char>(std::strtoul(byteHex, nullptr, 16));
    }
}