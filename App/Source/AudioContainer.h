#pragma once


struct AudioContainer{
    AudioContainer();
    ~AudioContainer();

    static const int NUM_SAMPLES;
    juce::Array<float> audio;
    bool playAudio;
    juce::Array<int> sampleIndex;
};