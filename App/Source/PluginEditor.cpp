#include "Headers.h"


//==============================================================================
AudioPluginAudioProcessorEditor::AudioPluginAudioProcessorEditor (AudioPluginAudioProcessor& p)
    : AudioProcessorEditor (&p), processorRef (p)
{
    juce::ignoreUnused (processorRef);

    // Create map and make it visible.
    m_Map.add(new Map(&processorRef.generator));
    addAndMakeVisible(m_Map[0]);

    // Main window.
    setSize(M_WINDOW_WIDTH, M_WINDOW_HEIGHT);
    setResizable(M_IS_WIDTH_RESIZABLE, M_IS_HEIGHT_RESIZABLE);
}

AudioPluginAudioProcessorEditor::~AudioPluginAudioProcessorEditor(){}

//==============================================================================
void AudioPluginAudioProcessorEditor::paint (juce::Graphics& g){}

void AudioPluginAudioProcessorEditor::resized(){
    auto r = getLocalBounds();
    auto mapArea = r;
    m_Map[0]->setBounds(mapArea);
}
