#include "Headers.h"

AudioContainer::AudioContainer(){
    Logger::writeToLog("AudioContainer created.");
}

AudioContainer::~AudioContainer(){
    Logger::writeToLog("AudioContainer destroyed.");
}

const int AudioContainer::NUM_SAMPLES = 131072;