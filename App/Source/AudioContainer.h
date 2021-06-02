#pragma once


struct AudioContainer{
    juce::Array<float> audio;
    bool playAudio;
    juce::Array<int> sampleIndex;
};