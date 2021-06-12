#pragma once


class AudioPluginAudioProcessorEditor: public juce::AudioProcessorEditor{
public:
    // Constructors and destructors.
    explicit AudioPluginAudioProcessorEditor (AudioPluginAudioProcessor&);
    ~AudioPluginAudioProcessorEditor() override;

    // View methods.
    void paint (juce::Graphics&) override;
    void resized() override;

private:
    // Member variables.
    AudioPluginAudioProcessor& processorRef;
    Map m_Map;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (AudioPluginAudioProcessorEditor)
};


