#pragma once


class AudioPluginAudioProcessorEditor: public juce::AudioProcessorEditor{
public:
    // Constructors and destructors.
    explicit AudioPluginAudioProcessorEditor (AudioPluginAudioProcessor&);
    ~AudioPluginAudioProcessorEditor() override;
    void setComponents();

    // View methods.
    void paint (juce::Graphics&) override;
    void resized() override;

private:
    // Member variables.
    AudioPluginAudioProcessor& m_ProcessorRef;
    
    TopBar m_TopBar;
    LeftBar m_LeftBar;
    juce::OwnedArray<Map> m_Maps;

    juce::DropShadow m_DropShadow;
    juce::DropShadower m_DropShadower;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (AudioPluginAudioProcessorEditor)
};


