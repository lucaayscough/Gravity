#include "Headers.h"


//------------------------------------------------------------//
// Constructors and destructors.

MapButton::MapButton(juce::OwnedArray<Map>& maps_ref)
    :   m_MapsRef(maps_ref){
    Logger::writeToLog("Created MapButton.");

    addAndMakeVisible(m_MapImage);
    m_MapImage.setInterceptsMouseClicks(false, false);
}

MapButton::~MapButton(){
    Logger::writeToLog("Destroyed MapButton.");
}

//------------------------------------------------------------//
// View methods.

void MapButton::paint(Graphics& g){
    juce::ignoreUnused(g);
}

void MapButton::resized(){
    Map& map = *(m_MapsRef[getButtonIndex()]);

    m_MapImage.setImage(map.createComponentSnapshot(map.getLocalBounds(), true, 0.1f));
    m_MapImage.setBounds(getLocalBounds().withTrimmedTop(Variables::LEFT_BAR_MAP_BOUNDARY).withTrimmedBottom(Variables::LEFT_BAR_MAP_BOUNDARY).withTrimmedLeft(Variables::LEFT_BAR_MAP_BOUNDARY));
}

//------------------------------------------------------------//
// Interface methods.

int MapButton::getButtonIndex(){return getComponentID().getIntValue();}

//------------------------------------------------------------//
// Controller methods.

void MapButton::mouseDown(const MouseEvent& e){
    juce::ignoreUnused(e);
    
    for(int i = 0; i < Variables::NUM_MAPS; i++){
        if(i == getButtonIndex()){
            m_MapsRef[i]->setVisible(true);
        }
        else{
            m_MapsRef[i]->setVisible(false);
        }
    }
}
