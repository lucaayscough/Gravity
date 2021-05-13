#include "PluginProcessor.h"
#include "PluginEditor.h"


//==============================================================================
AudioPluginAudioProcessorEditor::AudioPluginAudioProcessorEditor (AudioPluginAudioProcessor& p)
    : AudioProcessorEditor (&p), processorRef (p)
{
    juce::ignoreUnused (processorRef);
    // Make sure that before the constructor has finished, you've set the
    // editor's size to whatever you need it to be.

    // Main window.
    setSize (800, 400);
    setResizable(false, false);
    
    // Generate button.
    mGenerateButton.onClick = [&]() {processorRef.generator.generateSample(processorRef.generator.generateLatents());};
    addAndMakeVisible(mGenerateButton);
    addAndMakeVisible(mMap);
    addAndMakeVisible(mPlanet);
}

AudioPluginAudioProcessorEditor::~AudioPluginAudioProcessorEditor()
{
}

//==============================================================================
void AudioPluginAudioProcessorEditor::paint (juce::Graphics& g)
{
    g.fillAll(juce::Colours::lightgreen);
    
    // Generate button.
    mGenerateButton.setButtonText("BLaps");
}

void AudioPluginAudioProcessorEditor::resized()
{
    auto r = getLocalBounds();
    auto mapArea = r.reduced(30);
    mMap.setBounds(mapArea);
    mPlanet.setBounds(mapArea);
    mPlanet.setSize(50, 50);

    mGenerateButton.setBounds(
        (getWidth() - BUTTON_WIDTH) / 2,
        20,
        BUTTON_WIDTH,
        BUTTON_HEIGHT
    );
}
