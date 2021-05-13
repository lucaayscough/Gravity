#include "PluginProcessor.h"
#include "PluginEditor.h"

//==============================================================================
AudioPluginAudioProcessorEditor::AudioPluginAudioProcessorEditor (AudioPluginAudioProcessor& p)
    : AudioProcessorEditor (&p), processorRef (p)
{
    juce::ignoreUnused (processorRef);
    // Make sure that before the constructor has finished, you've set the
    // editor's size to whatever you need it to be.

    setSize (600, 600);
    
    mGenerateButton.onClick = [&]() {processorRef.generator.generateSample(processorRef.generator.generateLatents());};
    addAndMakeVisible(mGenerateButton);
}

AudioPluginAudioProcessorEditor::~AudioPluginAudioProcessorEditor()
{
}

//==============================================================================
void AudioPluginAudioProcessorEditor::paint (juce::Graphics& g)
{
    g.fillAll(juce::Colours::lightgreen);
    g.setColour(juce::Colours::orange);
    mGenerateButton.setButtonText("BLaps");
}

void AudioPluginAudioProcessorEditor::resized()

{
    mGenerateButton.setBounds(
        (getWidth() - BUTTON_WIDTH) / 2,
        (getHeight() - BUTTON_HEIGHT) / 2,
        BUTTON_WIDTH,
        BUTTON_HEIGHT
    );
}
