#pragma once

//==============================================================================
class AudioPluginAudioProcessorEditor  : public juce::AudioProcessorEditor
{
public:
    explicit AudioPluginAudioProcessorEditor (AudioPluginAudioProcessor&);
    ~AudioPluginAudioProcessorEditor() override;

    //==============================================================================
    void paint (juce::Graphics&) override;
    void resized() override;

private:
    // This reference is provided as a quick way for your editor to
    // access the processor object that created it.
    AudioPluginAudioProcessor& processorRef;
    
    // Window member variables.
    const int M_WINDOW_WIDTH = 1280;
    const int M_WINDOW_HEIGHT = 720;
    const bool M_IS_WIDTH_RESIZABLE = false;
    const bool M_IS_HEIGHT_RESIZABLE = false;
    
    // Planet member variables.
    const int M_DEFAULT_PLANET_DIAMETER = 50;

    // Main object inits.
    Map m_Map;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (AudioPluginAudioProcessorEditor)
};


