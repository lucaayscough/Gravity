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
    AudioPluginAudioProcessor& processorRef;
    Map m_Map;

    // Window member variables.
    const int M_WINDOW_WIDTH = Variables::WINDOW_WIDTH;
    const int M_WINDOW_HEIGHT = Variables::WINDOW_HEIGHT;
    const bool M_IS_WIDTH_RESIZABLE = Variables::IS_WIDTH_RESIZABLE;
    const bool M_IS_HEIGHT_RESIZABLE = Variables::IS_HEIGHT_RESIZABLE;
    
    // Planet member variables.
    const int M_DEFAULT_PLANET_DIAMETER = Variables::DEFAULT_PLANET_DIAMETER;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (AudioPluginAudioProcessorEditor)
};


