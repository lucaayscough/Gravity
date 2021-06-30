#include "Headers.h"


//------------------------------------------------------------//
// Constructors and destructors.

AudioPluginAudioProcessorEditor::AudioPluginAudioProcessorEditor(AudioPluginAudioProcessor& p)
    :   AudioProcessorEditor(&p),
        processorRef(p),
        m_Map(processorRef.m_AudioContainer, processorRef.m_Parameters),
        m_DropShadow(Variables::TOP_BAR_SHADOW_COLOUR, 10, juce::Point<int>({0, 0})), m_DropShadower(m_DropShadow){
    Logger::writeToLog("Editor created.");

    addAndMakeVisible(m_TopBar);
    addAndMakeVisible(m_LeftBar);
    addAndMakeVisible(m_Map);

    m_TopBar.setAlwaysOnTop(true);

    m_DropShadower.setOwner(&m_TopBar);

    // Main window.
    setSize(Variables::WINDOW_WIDTH, Variables::WINDOW_HEIGHT);
    setResizable(Variables::IS_WIDTH_RESIZABLE, Variables::IS_HEIGHT_RESIZABLE);
}

AudioPluginAudioProcessorEditor::~AudioPluginAudioProcessorEditor(){
    Logger::writeToLog("Editor destroyed.");
}

//------------------------------------------------------------//
// View methods.

void AudioPluginAudioProcessorEditor::paint(juce::Graphics& g){
    g.fillAll(Variables::EDITOR_BG_COLOUR);
}

void AudioPluginAudioProcessorEditor::resized(){
    auto r = getLocalBounds();
    
    auto topBar = r.removeFromTop(Variables::TOP_BAR);
    m_TopBar.setBounds(topBar);

    auto leftBar = r.removeFromLeft(Variables::LEFT_BAR);
    m_LeftBar.setBounds(leftBar);

    auto mapArea = r.withTrimmedRight(Variables::MAP_TRIM).withTrimmedBottom(Variables::MAP_TRIM);
    m_Map.setBounds(mapArea);
}
