
/**
 * Interface.h
 *
 * In this header file, we implement a
 * basic User Interface using the Command
 * Line Window. The defined functions are
 * used to print information about the data
 * processed by the neural network, about
 * the dataset, about the neural network's
 * progress and help the user understand
 * how to use the project.
 */

#pragma once

#include "Common.hpp"

#ifdef _WIN32
#define _CRT_SECURE_NO_WARNINGS 1
#include <windows.h>
#else
#include <termios.h>
#include <unistd.h>
#endif

void setupConsole(void);
void restoreConsole(void);
void getCursorPosition(int* row, int* col);
void usage(char* filename);
void print_epoch_stats(int epoch, double epoch_loss, int epoch_accuracy, double benchmark, int des_x, int des_y);

void clearScreen(void);
void gotoxy(int x, int y);
void hideCursor(void);
void showCursor(void);
void saveCursorPosition(void);
void restoreCursorPosition(void);

class progress_bar
{
public:
    std::string message;
    char* bar;
    char progress_token;
    int progress;
    int length;

    void indicate_progress(double checkpoint, int x, int y);

    progress_bar(std::string message, char progress_token, int length) :
        message{ message },
        progress_token{ progress_token },
        length{ length }
    {
        bar = new char[length + 1];
        for (int i = 0; i < length; i += 1)
        {
            bar[i] = ' ';
        }
        bar[length] = '\0';
        progress = 0;
        std::cout << "\n";
    }

    ~progress_bar()
    {
        delete[] bar;
    }
};
