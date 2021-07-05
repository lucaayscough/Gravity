#include "Headers.h"


//--------------------------------------------------//
// Constructors and destructors.

Planet::Planet(juce::String& id, juce::OwnedArray<Planet>& planets, AudioContainer& audiocontainer, Parameters& parameters, ControlPanel& controlpanel)
    :   Astro(id, audiocontainer, parameters, controlpanel),
        m_PlanetsRef(planets){
    setGradients();
    setListeners();
}

Planet::~Planet(){
    m_Animator.m_AreaShift.removeListener(this);
}

void Planet::setGradients(){
    m_ColourGradient.addColour((double)0.0, juce::Colours::white);
    m_ColourGradient.addColour((double)0.2, juce::Colours::yellow);
    m_ColourGradient.addColour((double)0.4, juce::Colours::orange);
    m_ColourGradient.addColour((double)0.7, juce::Colours::red);
    m_ColourGradient.addColour((double)1.0, juce::Colours::darkred);
}

//--------------------------------------------------//
// Init methods.

void Planet::setListeners(){
    m_Animator.m_AreaShift.addListener(this);
}

//--------------------------------------------------//
// View methods.

void Planet::paint(Graphics& g){
    if(getState().getProperty(Parameters::isActiveProp)){
        g.setColour(juce::Colours::green);
    }
    else{
        double max_distance = sqrt((double)(pow(getParentWidth() / 2, 2)) + (double)(pow(getParentHeight() / 2, 2)));
        double pos = (getDistance(getCentreX(), getCentreY(), getParentWidth() / 2, getParentHeight() / 2)) / max_distance;
        
        g.setColour(m_ColourGradient.getColourAtPosition(pos));
    }

    float shift = m_Animator.getShiftedDiameter(getArea()) - getDiameter();

    g.fillEllipse(
        (float)getClipBoundary() / 2.0f - shift / 2.0f,
        (float)getClipBoundary() / 2.0f - shift / 2.0f,
        (float)getDiameter() + shift,
        (float)getDiameter() + shift
    );
}

void Planet::resized(){
    draw(getDiameter(), getX(), getY());
    setPosXY(getX(), getY());
}

void Planet::resizePlanet(float area){
    int new_diameter = (int)(round(sqrt(area / 3.1415f)) * 2.0f);
    int old_diameter = getDiameter();
    int diff = old_diameter - new_diameter;
    
    setArea(area);
    
    draw(
        (int)new_diameter,
        getX() + (int)diff / 2,
        getY() + (int)diff / 2
    );

    // TODO:
    // NEED A WAY TO UPDATE GRAPH WHEN DONE ZOOMING IN ON SOUND
    //mixLatents();
}

void Planet::checkCollision(){
    int centrePosX = getX() + getRadiusWithClipBoundary();
    int centrePosY = getY() + getRadiusWithClipBoundary();

    float distance, minDistance;

    // Check collision with sun.
    {
        int centreXSun = getParentWidth() / 2;
        int centreYSun = getParentHeight() / 2;
        
        int sunRadius = (int)sqrt(Variables::SUN_AREA / 3.1415f);

        distance = getDistance(centrePosX, centrePosY, centreXSun, centreYSun);
        minDistance = sunRadius + getRadius();

        if(distance <= minDistance){
            draw(getDiameter(), getPosX(), getPosY());
        }
    }

    Planet* planet;
    int centrePosX2, centrePosY2;

    for(int i = 0; i < m_PlanetsRef.size(); i++){
        // Variable for ease of use.
        planet = m_PlanetsRef[i];

        // Avoid self collision testing.
        if(planet->getComponentID() != getComponentID()){
            centrePosX2 = planet->getX() + planet->getRadiusWithClipBoundary();
            centrePosY2 = planet->getY() + planet->getRadiusWithClipBoundary();

            distance = getDistance(centrePosX, centrePosY, centrePosX2, centrePosY2);
            minDistance = planet->getRadius() + getRadius();

            if(distance <= minDistance){
                draw(getDiameter(), getPosX(), getPosY());
            }
        }
    }
}

void Planet::checkBounds(){
    //Check left boundary.
    if(getX() < -(getClipBoundary() / 2))
        draw(getDiameter(), -(getClipBoundary() / 2), getY());

    // Check top boundary.
    if(getY() < -(getClipBoundary() / 2))
        draw(getDiameter(), getX(), -(getClipBoundary() / 2));

    // Check right boundary,
    if(getX() + getDiameter() + (getClipBoundary() / 2) > getParentWidth())
        draw(getDiameter(), getParentWidth() - getDiameter() - (getClipBoundary() / 2), getY());

    // Check bottom boundary.
    if(getY() + getDiameter() + (getClipBoundary() / 2) > getParentHeight())
        draw(getDiameter(), getX(), getParentHeight() - getDiameter() - (getClipBoundary() / 2));
}

//--------------------------------------------------//
// Interface methods.

juce::ValueTree Planet::getState(){
    auto id = getParentComponent()->getComponentID();
    auto mapNode = m_ParametersRef.getMapNode(id);
    return m_ParametersRef.getRootPlanetNode(mapNode).getChildWithProperty(Parameters::idProp, getComponentID());
}

//--------------------------------------------------//
// Controller methods.

void Planet::mouseDown(const MouseEvent& e){m_Dragger.startDraggingComponent(this, e);}

void Planet::mouseUp(const MouseEvent& e){
    if(e.mods.isLeftButtonDown()){
        // Generates new sample if double clicked with left mouse button.
        if(e.getNumberOfClicks() > 1)
            generateSample();
        
        // Plays sample if clicked once with left mouse button.
        else if(e.getNumberOfClicks() == 1 && e.mouseWasClicked())
            playSample();

        // Updates latent mixture graph if there has been a dragging motion.
        else if(e.mouseWasDraggedSinceMouseDown())
            m_ParametersRef.mixLatents(getMapNode());
    }
    
    // Destroys planet if clicked with right mouse button.
    else if(e.mods.isRightButtonDown())
        m_ParametersRef.removePlanetNode(getState());
}

void Planet::mouseDrag(const MouseEvent& e){
    m_Dragger.dragComponent(this, e, nullptr);
    checkCollision();
    checkBounds();
    setPosXY(getX(), getY());
}

void Planet::mouseWheelMove(const MouseEvent& e, const MouseWheelDetails& w){
    juce::ignoreUnused(e);

    if(w.deltaY > 0.0f && getArea() + Variables::AREA_MODIFIER <= Variables::MAX_PLANET_AREA)
        resizePlanet(getArea() + Variables::AREA_MODIFIER);

    else if(w.deltaY < 0.0f && getArea() - Variables::AREA_MODIFIER >= Variables::MIN_PLANET_AREA)
        resizePlanet(getArea() - Variables::AREA_MODIFIER);

}
