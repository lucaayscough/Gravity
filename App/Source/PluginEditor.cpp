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
    setSize (1280, 720);
    setResizable(false, false);
    
    // Generate button.
    mGenerateButton.onClick = [&]() {processorRef.generator.generateSample(processorRef.generator.generateLatents());};
    
    mSun.getNewSample = [&]() {
        processorRef.generator.generateSample(
            processorRef.generator.generateLatents()
        );
    };

    addAndMakeVisible(mGenerateButton);
    addAndMakeVisible(mMap);
    addAndMakeVisible(mSun);
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
    auto mapArea = r;
    mMap.setBounds(mapArea);
    mSun.setBounds(
        (getWidth() - mSun.DIAMETER) / 2,
        (getHeight() - mSun.DIAMETER) / 2,
        mSun.DIAMETER,
        mSun.DIAMETER
    );
    mPlanet.setBounds(mapArea);

    mGenerateButton.setBounds(
        (getWidth() - BUTTON_WIDTH) / 2,
        20,
        BUTTON_WIDTH,
        BUTTON_HEIGHT
    );
}
