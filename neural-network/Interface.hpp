
/**
 * Interface.hpp
 *
 * In this header file, we implement a
 * basic User Interface using the Command
 * Line Window. The defined functions are
 * used to print information about the data
 * processed by the neural network, about
 * the dataset, about the neural network's
 * progress and help the user understand
 * how to use the project. there is also a 
 * cross platform implementation of a progress
 * bar.
 * 
 * @remark https://github.com/sol-prog/ansi-escape-codes-windows-posix-terminals-c-programming-examples
 */

#pragma once

#include "Common.hpp"

#ifdef _WIN32
#define _CRT_SECURE_NO_WARNINGS 1
#include <windows.h>
#else
#include <sys/ioctl.h>
#include <termios.h>
#include <unistd.h>
#endif

void setupConsole(void);
void restoreConsole(void);
void getWindowSize(int(&rows), int(&columns));
void getCursorPosition(int* row, int* col);
void usage(char* filename);
void print_epoch_stats(int epoch, double epoch_loss, int epoch_accuracy, double benchmark);

void SetConsoleWindowSize(int x, int y);
void moveUp(int positions);
void moveDown(int positions);
void scrollUp(int positions);
void scrollDown(int positions);
void clearScreen(void);
void gotoxy(int x, int y);
void hideCursor(void);
void showCursor(void);
void saveCursorPosition(void);
void restoreCursorPosition(void);


/**
 * Implements a progress bar in CLI.
 * 
 * Instances of this class are progress bars 
 * that inform the user of a large task's 
 * progress. There is also a short description
 * attached to each progress bar. The description 
 * is recommended not to exceed the length of 35
 * characters.
 */
class progress_bar
{
public:
    std::string message;
    char* bar;
    char progress_token;
    int progress;
    int length;

    void indicate_progress(double checkpoint);

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
