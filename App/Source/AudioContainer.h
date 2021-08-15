#pragma once


struct AudioContainer{
    AudioContainer();
    ~AudioContainer();

    static const int M_NUM_SAMPLES;
    juce::Array<float> m_Audio;
    bool m_PlayAudio;
    juce::Array<int> m_SampleIndex;
};