#include "Headers.h"


AudioPluginAudioProcessorEditor::AudioPluginAudioProcessorEditor(AudioPluginAudioProcessor& p):
    AudioProcessorEditor(&p),
    processorRef(p),
    m_Map(&processorRef.m_AudioContainer, processorRef.m_Parameters){
    Logger::writeToLog("Construct editor.");
    addAndMakeVisible(m_Map);

    // Main window.
    setSize(M_WINDOW_WIDTH, M_WINDOW_HEIGHT);
    setResizable(M_IS_WIDTH_RESIZABLE, M_IS_HEIGHT_RESIZABLE);
}

AudioPluginAudioProcessorEditor::~AudioPluginAudioProcessorEditor(){}

//==============================================================================
void AudioPluginAudioProcessorEditor::paint(juce::Graphics& g){juce::ignoreUnused(g);}

void AudioPluginAudioProcessorEditor::resized(){
    auto r = getLocalBounds();
    auto mapArea = r;
    m_Map.setBounds(mapArea);
}
