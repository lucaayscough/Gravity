#include "Headers.h"


//------------------------------------------------------------//
// Constructors and destructors.

LeftBar::LeftBar(juce::OwnedArray<Map>& maps_ref)
    :   m_MapsRef(maps_ref){
    Logger::writeToLog("Created LeftBar.");
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

    auto r = getLocalBounds().withTrimmedTop(Variables::LEFT_BAR_TOP_BOUNDARY).withTrimmedBottom(Variables::LEFT_BAR_BOTTOM_BOUNDARY);
    auto button_height = r.getHeight() / Variables::NUM_MAPS;
}
