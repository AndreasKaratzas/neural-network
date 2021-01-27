
#include "Interface.hpp"

#ifdef _WIN32

#ifndef ENABLE_VIRTUAL_TERMINAL_PROCESSING
#define ENABLE_VIRTUAL_TERMINAL_PROCESSING 0x0004
#endif

static HANDLE stdoutHandle, stdinHandle;
static DWORD outModeInit, inModeInit;

void setupConsole(void)
{
    DWORD outMode = 0, inMode = 0;
    stdoutHandle = GetStdHandle(STD_OUTPUT_HANDLE);
    stdinHandle = GetStdHandle(STD_INPUT_HANDLE);

    if (stdoutHandle == INVALID_HANDLE_VALUE || stdinHandle == INVALID_HANDLE_VALUE)
    {
        exit(GetLastError());
    }

    if (!GetConsoleMode(stdoutHandle, &outMode) || !GetConsoleMode(stdinHandle, &inMode))
    {
        exit(GetLastError());
    }

    outModeInit = outMode;
    inModeInit = inMode;

    // Enable ANSI escape codes
    outMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;

    // Set stdin as no echo and unbuffered
    inMode &= ~(ENABLE_ECHO_INPUT | ENABLE_LINE_INPUT);

    if (!SetConsoleMode(stdoutHandle, outMode) || !SetConsoleMode(stdinHandle, inMode))
    {
        exit(GetLastError());
    }
}

void restoreConsole(void)
{
    // Reset colors
    printf("\x1b[0m");

    // Reset console mode
    if (!SetConsoleMode(stdoutHandle, outModeInit) || !SetConsoleMode(stdinHandle, inModeInit))
    {
        exit(GetLastError());
    }
}
#else

static struct termios orig_term;
static struct termios new_term;

void setupConsole(void)
{
    tcgetattr(STDIN_FILENO, &orig_term);
    new_term = orig_term;

    new_term.c_lflag &= ~(ICANON | ECHO);

    tcsetattr(STDIN_FILENO, TCSANOW, &new_term);
}

void restoreConsole(void)
{
    // Reset colors
    printf("\x1b[0m");

    // Reset console mode
    tcsetattr(STDIN_FILENO, TCSANOW, &orig_term);
}
#endif

void getCursorPosition(int* row, int* col)
{
    printf("\x1b[6n");
    char buff[128];
    int indx = 0;
    for (;;)
    {
        int cc = getchar();
        buff[indx] = (char)cc;
        indx++;
        if (cc == 'R')
        {
            buff[indx + 1] = '\0';
            break;
        }
    }
    sscanf(buff, "\x1b[%d;%dR", row, col);
    fseek(stdin, 0, SEEK_END);
}

void clearScreen(void)
{
    printf("\033[2J");
}

void gotoxy(int x, int y)
{
    printf("\x1b[%d;%df", x, y);
}

void hideCursor(void)
{
    printf("\x1b[?25l");
}

void showCursor(void)
{
    printf("\x1b[?25h");
}

void saveCursorPosition(void)
{
    printf("\x1b%d", 7);
}

void restoreCursorPosition(void)
{
    printf("\x1b%d", 8);
}

/**
 * Indicates progress of an operation.
 *
 * @param[in] percentage completion rate of the monitored task
 */
void progress_bar::indicate_progress(double checkpoint, int x, int y)
{
    gotoxy(x, y);
    std::cout << "\r" << message << "\t|";
    int ratio = (int)std::ceil(checkpoint * length);
    int completion_percentage = (int)std::ceil(checkpoint * 100);
    if (ratio > length)
    {
        ratio -= 1;
    }
    if (completion_percentage > 100)
    {
        completion_percentage -= 1;
    }
    for (int i = 0; i < ratio; i += 1)
    {
        bar[i] = progress_token;
    }
    std::cout << bar << "| " << std::setw(4) << completion_percentage << "%";
}


/**
 * Prints epoch stats. More specifically, it prints the epoch's number
 * along with the model's acuracy and loss. It also prints the epoch's benchmark.
 *
 * @param[in] epoch the epoch's number
 * @param[in] epoch_loss the model's loss during a certain epoch of training or evaluation
 * @param[in] epoch_accuracy the model's accuracy during a certain epoch of training or evaluation
 * @param[in] benchmark the epoch's benchmark
 */
void print_epoch_stats(int epoch, double epoch_loss, int epoch_accuracy, double benchmark, int des_x, int des_y)
{
    gotoxy(des_x, des_y);
    if (epoch == 0)
    {
        std::cout << "\n";
    }
    if (epoch == -1)
    {
        std::cout << "\n\n[EVALUATION] [LOSS " << std::fixed << std::setprecision(5) << epoch_loss << "] [ACCURACY " << std::setw(6) << epoch_accuracy << " out of " << (int)MNIST_TEST << "] Work took " << std::setw(4) << (int)benchmark << " seconds";
    }
    else
    {
        std::cout << "\n[EPOCH " << std::setw(4) << epoch << "] [LOSS " << std::fixed << std::setprecision(5) << epoch_loss << "] [ACCURACY " << std::setw(6) << epoch_accuracy << " out of " << (int)MNIST_TRAIN << "] Work took " << std::setw(4) << (int)benchmark << " seconds";
    }
}

/**
 * Prints information regarding the usage and the available options of the project.
 *
 * @param[in] filename the filepath of the executable
 *
 * @note Upon this function's execution, the program is terminated.
 */
void usage(char* filename)
{
    std::cout << "Usage of " << filename << ":\n";
    std::cout << "\t:option \'-i\': integer \t - \t The size of the input layer for the neural network.\n";
    std::cout << "\t:option \'-h\': integer \t - \t The size of a hidden layer for the neural network.\n\t\t\t\t\t There can be multiple hidden layers. For every hidden layer, use this option.\n";
    std::cout << "\t:option \'-o\': integer \t - \t The size of the output layer for the neural network.\n";
    exit(8);
}
