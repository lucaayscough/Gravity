#pragma once


class AudioPluginAudioProcessor: public juce::AudioProcessor, juce::ValueTree::Listener{
public:
    //==============================================================================
    AudioPluginAudioProcessor();
    ~AudioPluginAudioProcessor() override;

    //==============================================================================
    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;
    bool isBusesLayoutSupported (const BusesLayout& layouts) const override;
    void processBlock (juce::AudioBuffer<float>&, juce::MidiBuffer&) override;
    using AudioProcessor::processBlock;
    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override;
    const juce::String getName() const override;
    bool acceptsMidi() const override;
    bool producesMidi() const override;
    bool isMidiEffect() const override;
    double getTailLengthSeconds() const override;
    int getNumPrograms() override;
    int getCurrentProgram() override;
    void setCurrentProgram (int index) override;
    const juce::String getProgramName (int index) override;
    void changeProgramName (int index, const juce::String& newName) override;
    void getStateInformation(juce::MemoryBlock& destData) override;
    void setStateInformation(const void* data, int sizeInBytes) override;
    void rebuildState();
    void valueTreePropertyChanged(juce::ValueTree&, const juce::Identifier&) override;

    //==============================================================================

    Generator m_Generator;
    AudioContainer m_AudioContainer;

    // ValueTreeState variables.
    juce::AudioProcessorValueTreeState m_ValueTreeState;
    Parameters m_Parameters;

private:
    void playAudio(juce::AudioBuffer<float>&, int, int);
    void stopAudio();
    void playSample();
    void addSample(juce::ValueTree);

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (AudioPluginAudioProcessor)
};
