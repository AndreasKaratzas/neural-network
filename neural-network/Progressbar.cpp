
#include "Progressbar.h"

/**
 * Indicates progress of an operation.
 * 
 * @param[in] percentage completion rate of the monitored task
 * @param[in] width the total width of the progress bar
 * @param[in] overlay the completion rate transformed in characters to be displayed in the progressbar
 * @param[in] message a short description of the monitored task
 */
void printProgress(long double percentage, int width, char* overlay, char* message)
{
    int val = (int)ceil(percentage * 100);
    int lpad = (int)ceil(percentage * width);
    int rpad = width - lpad;
    printf("\r%s\t[%.*s%*s] %3d%%", message, lpad, overlay, rpad, "", val);
    fflush(stdout);
}
