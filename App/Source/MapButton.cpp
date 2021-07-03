#include "Headers.h"


//------------------------------------------------------------//
// Constructors and destructors.

MapButton::MapButton(juce::OwnedArray<Map>& maps_ref, const juce::String& id)
    :   m_MapsRef(maps_ref){
    setComponentID(id);
    addAndMakeVisible(m_MapImage);
    m_MapImage.setInterceptsMouseClicks(false, false);
}

MapButton::~MapButton(){}

//------------------------------------------------------------//
// Init methods.

void MapButton::setListeners(){
    getMap().m_UpdateImage.addListener(this);
}

//------------------------------------------------------------//
// View methods.

void MapButton::paint(Graphics& g){
    if(getMap().isVisible())
        g.fillAll(juce::Colours::white);
    else
        g.fillAll(juce::Colours::black);
}

void MapButton::resized(){
    setImage();
    m_MapImage.setBounds(getLocalBounds().withTrimmedTop(Variables::LEFT_BAR_MAP_BOUNDARY).withTrimmedBottom(Variables::LEFT_BAR_MAP_BOUNDARY).withTrimmedLeft(Variables::LEFT_BAR_MAP_BOUNDARY));
}

//------------------------------------------------------------//
// Interface methods.

int MapButton::getButtonIndex(){return getComponentID().getIntValue();}
Map& MapButton::getMap(){return *m_MapsRef[getButtonIndex()];}

void MapButton::setImage(){
    Map& map = getMap();
    m_MapImage.setImage(map.createComponentSnapshot(map.getLocalBounds(), true, 0.1f));
    getMap().m_UpdateImage = false;
}

//------------------------------------------------------------//
// Controller methods.

void MapButton::mouseDown(const MouseEvent& e){
    juce::ignoreUnused(e);
    
    for(int i = 0; i < Variables::NUM_MAPS; i++){
        if(i == getButtonIndex()){
            m_MapsRef[i]->setVisible(true);
            m_MapsRef[i]->setInterceptsMouseClicks(true, true);
        }
        else{
            m_MapsRef[i]->setVisible(false);
            m_MapsRef[i]->setInterceptsMouseClicks(false, false);
        }
    }

    getParentComponent()->repaint();
}

//------------------------------------------------------------//
// Callback methods.

void MapButton::valueChanged(juce::Value& v){
    juce::ignoreUnused(v);
    setImage(); 
}
