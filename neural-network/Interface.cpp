
#include "Interface.hpp"

/**
 * Includes graphic libraries depending on the host's OS.
 * Sets up CLI environment and defines functions and modules
 * for the CLI.
 * 
 * @remark https://github.com/sol-prog/ansi-escape-codes-windows-posix-terminals-c-programming-examples
 */
#ifdef _WIN32

#ifndef ENABLE_VIRTUAL_TERMINAL_PROCESSING
#define ENABLE_VIRTUAL_TERMINAL_PROCESSING 0x0004
#endif

static HANDLE stdoutHandle, stdinHandle;
static DWORD outModeInit, inModeInit;

/**
 * Sets up console (CLI) in Windows OS.
 * 
 * @remark https://github.com/sol-prog/ansi-escape-codes-windows-posix-terminals-c-programming-examples
 */
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
    outMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;                      /// Enable ANSI escape codes
    inMode &= ~(ENABLE_ECHO_INPUT | ENABLE_LINE_INPUT);                 /// Set stdin as no echo and unbuffered

    if (!SetConsoleMode(stdoutHandle, outMode) || !SetConsoleMode(stdinHandle, inMode))
    {
        exit(GetLastError());
    }
}

/**
 * Resets console (CLI) in Windows OS. During
 * the execution there might be some changes 
 * in the CLI which need to be unmade upon 
 * program's termination.
 * 
 * @remark https://github.com/sol-prog/ansi-escape-codes-windows-posix-terminals-c-programming-examples
 */
void restoreConsole(void)
{
    printf("\x1b[0m");                                                  /// Reset colors

    if (!SetConsoleMode(stdoutHandle, outModeInit) || !SetConsoleMode(stdinHandle, inModeInit))
    {                                                                   /// Reset console mode
        exit(GetLastError());
    }
}
#else

static struct termios orig_term;
static struct termios new_term;

/**
 * Sets up console (CLI) in Unix OS.
 * 
 * @remark https://github.com/sol-prog/ansi-escape-codes-windows-posix-terminals-c-programming-examples
 */
void setupConsole(void)
{
    tcgetattr(STDIN_FILENO, &orig_term);
    new_term = orig_term;

    new_term.c_lflag &= ~(ICANON | ECHO);

    tcsetattr(STDIN_FILENO, TCSANOW, &new_term);
}

/**
 * Resets console (CLI) in Unix OS. During
 * the execution there might be some changes
 * in the CLI which need to be unmade upon
 * program's termination.
 * 
 * @remark https://github.com/sol-prog/ansi-escape-codes-windows-posix-terminals-c-programming-examples
 */
void restoreConsole(void)
{
    printf("\x1b[0m");                                                  /// Reset colors
    tcsetattr(STDIN_FILENO, TCSANOW, &orig_term);                       /// Reset console mode
}
#endif

/**
 * Fetches cursor position with respect to the CLI.
 * 
 * @param[in, out] row the row of the cursor in the CLI
 * @param[in, out] col the column of the cursor in the CLI
 * 
 * @remark https://github.com/sol-prog/ansi-escape-codes-windows-posix-terminals-c-programming-examples
 */
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

/**
 * Clears CLI screen
 * 
 * @remark https://github.com/sol-prog/ansi-escape-codes-windows-posix-terminals-c-programming-examples
 */
void clearScreen(void)
{
    printf("\033[2J");
}

/**
 * Places CLI cursor at defined position in the CLI.
 * 
 * @param[in] x the row to place the cursor
 * @param[in] y the column to place the cursor
 * 
 * @remark https://github.com/sol-prog/ansi-escape-codes-windows-posix-terminals-c-programming-examples
 * 
 * @remark https://stackoverflow.com/questions/10401724/move-text-cursor-to-particular-screen-coordinate
 */
void gotoxy(int x, int y)
{
    printf("\x1b[%d;%df", x, y);
}

/**
 * Hides the cursor from the CLI.
 * 
 * @remark https://github.com/sol-prog/ansi-escape-codes-windows-posix-terminals-c-programming-examples
 */
void hideCursor(void)
{
    printf("\033[?25l");
}

/**
 * Makes CLI cursor visible.
 * 
 * @remark https://github.com/sol-prog/ansi-escape-codes-windows-posix-terminals-c-programming-examples
 */
void showCursor(void)
{
    printf("\033[?25h");
}

/**
 * Saves cursor's position in the CLI.
 *
 * @remark https://github.com/sol-prog/ansi-escape-codes-windows-posix-terminals-c-programming-examples
 */
void saveCursorPosition(void)
{
    printf("\x1b%d", 7);
}

/**
 * Restores cursor's position in the CLI.
 *
 * @remark https://github.com/sol-prog/ansi-escape-codes-windows-posix-terminals-c-programming-examples
 */
void restoreCursorPosition(void)
{
    printf("\x1b%d", 8);
}

/**
 * Indicates progress of an operation.
 *
 * @param[in] percentage completion rate of the monitored task
 * @param[in] x the row of the CLI to print the progess bar
 * @param[in] y the column of the CLI to print the progess bar
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
 * @param[in] des_x the row of the CLI to print the epoch stats message
 * @param[in] des_y the column of the CLI to print the epoch stats message
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
