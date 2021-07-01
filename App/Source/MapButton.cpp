#include "Headers.h"


//------------------------------------------------------------//
// Constructors and destructors.

MapButton::MapButton(juce::OwnedArray<Map>& maps_ref)
    :   m_MapsRef(maps_ref){
    Logger::writeToLog("Created MapButton.");

    addAndMakeVisible(m_MapImage);
}

MapButton::~MapButton(){
    Logger::writeToLog("Destroyed MapButton.");
}

//------------------------------------------------------------//
// View methods.

void MapButton::paint(Graphics& g){}

void MapButton::resized(){
    Map& map = *(m_MapsRef[getButtonIndex()]);

    auto image = map.createComponentSnapshot(map.getLocalBounds(), true, 0.1f);
    m_MapImage.setImage(image);
    m_MapImage.setBounds(getLocalBounds());
}

//------------------------------------------------------------//
// Interface methods.

int MapButton::getButtonIndex(){return getComponentID().getIntValue();}
