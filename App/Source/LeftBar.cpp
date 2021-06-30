#include "Headers.h"


//------------------------------------------------------------//
// Constructors and destructors.

LeftBar::LeftBar(){
    Logger::writeToLog("Created LeftBar.");

    addAndMakeVisible(m_MapButton1);
    addAndMakeVisible(m_MapButton2);
    addAndMakeVisible(m_MapButton3);
    addAndMakeVisible(m_MapButton4);
    addAndMakeVisible(m_MapButton5);
    addAndMakeVisible(m_MapButton6);
    addAndMakeVisible(m_MapButton7);
    addAndMakeVisible(m_MapButton8);
}

LeftBar::~LeftBar(){
    Logger::writeToLog("Destroyed LeftBar.");
}

//------------------------------------------------------------//
// View methods.

void LeftBar::paint(Graphics& g){
    g.fillAll(Variables::EDITOR_BG_COLOUR);
}

void LeftBar::resized(){
    // TODO:
    // Clean this up.

    auto r = getLocalBounds().withTrimmedTop(20).withTrimmedBottom(20);
    auto button_height = r.getHeight() / 8;

    auto map_button_1 = r.removeFromTop(button_height);
    m_MapButton1.setBounds(map_button_1);

    auto map_button_2 = r.removeFromTop(button_height);
    m_MapButton2.setBounds(map_button_2);

    auto map_button_3 = r.removeFromTop(button_height);
    m_MapButton3.setBounds(map_button_3);

    auto map_button_4 = r.removeFromTop(button_height);
    m_MapButton4.setBounds(map_button_4);

    auto map_button_5 = r.removeFromTop(button_height);
    m_MapButton5.setBounds(map_button_5);

    auto map_button_6 = r.removeFromTop(button_height);
    m_MapButton6.setBounds(map_button_6);

    auto map_button_7 = r.removeFromTop(button_height);
    m_MapButton7.setBounds(map_button_7);

    auto map_button_8 = r.removeFromTop(button_height);
    m_MapButton8.setBounds(map_button_8);
}
