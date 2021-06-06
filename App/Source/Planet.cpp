#include "Headers.h"


// Main Planet class.

//--------------------------------------------------//
// Constructors and destructors.

Planet::Planet(){}

Planet::Planet(juce::OwnedArray<Planet>* planets_ptr, AudioContainer* audiocontainer_ptr, juce::ValueTree state):
    m_PlanetsPtr(planets_ptr),
    m_AudioContainerPtr(audiocontainer_ptr),
    m_State(state){
    
    // Listener value used to determine when to destroy the planet.
    m_Destroy.setValue(false);

    // Listener used to detect when lerp graph needs recalculating.
    m_LerpGraph.setValue(false);

    setPosXY(getX(), getY());
}

Planet::~Planet(){}

//--------------------------------------------------//
// View methods.

void Planet::paint(Graphics& g){
    g.setColour(juce::Colours::red);
    draw(getDiameter(), getX(), getY());
    g.fillEllipse(getClipBoundary() / 2, getClipBoundary() / 2, getDiameter(), getDiameter());
}

void Planet::resized(){}

void Planet::draw(int diameter, int x, int y){
    setBounds(x, y, diameter + getClipBoundary(), diameter + getClipBoundary());
}

void Planet::resizePlanet(int diameter){
    int new_x;
    int new_y;

    if(diameter > getDiameter()){
        new_x = getX() - (Variables::SIZE_MODIFIER / 2);
        new_y = getY() - (Variables::SIZE_MODIFIER / 2);
    } else{
        new_x = getX() + (Variables::SIZE_MODIFIER / 2);
        new_y = getY() + (Variables::SIZE_MODIFIER / 2);
    }

    setDiameter(diameter);
    draw(diameter, new_x, new_y);

    // TODO:
    // NEED A WAY TO UPDATE GRAPH WHEN DONE ZOOMING IN ON SOUND
    //updateGraph();
}

void Planet::setDiameter(int diameter){m_State.setProperty(Parameters::diameterProp, diameter, nullptr);}
void Planet::setPosXY(int x, int y){
    m_State.setProperty(Parameters::posXProp, x, nullptr);
    m_State.setProperty(Parameters::posYProp, y, nullptr);
}

int Planet::getDiameter(){return m_State.getProperty(Parameters::diameterProp);}
int Planet::getPosX(){return m_State.getProperty(Parameters::posXProp);}
int Planet::getPosY(){return m_State.getProperty(Parameters::posYProp);}
int Planet::getMapWidth(){return m_State.getProperty(Parameters::mapWidthProp);}
int Planet::getMapHeight(){return m_State.getProperty(Parameters::mapHeightProp);}
int Planet::getClipBoundary(){return Variables::CLIP_BOUNDARY;}

float Planet::getDistance(int xa, int ya, int xb, int yb){  
    float a = (float)pow(xb - xa, 2);
    float b = (float)pow(yb - ya, 2); 
    return sqrt(a + b);
}

float Planet::getDistance(Planet* planet_a, Planet* planet_b){  
    int centreXA = getCentreX(planet_a);
    int centreYA = getCentreY(planet_a);
    int centreXB = getCentreX(planet_b);
    int centreYB = getCentreY(planet_b);

    float a = (float)pow(centreXB - centreXA, 2);
    float b = (float)pow(centreYB - centreYA, 2);

    return sqrt(a + b);
}

int Planet::getCentreX(Planet* planet){
    return planet->getX() + ((planet->getDiameter() + planet->getClipBoundary()) / 2);
}

int Planet::getCentreY(Planet* planet){
    return planet->getY() + ((planet->getDiameter() + planet->getClipBoundary()) / 2);
}

void Planet::updateGraph(){
    m_LerpGraph.setValue(true);
    m_LerpGraph.setValue(false);
}

void Planet::addSample(){
    m_AudioContainerPtr->audio.clear();

    juce::Array<float> sample;
    sample.ensureStorageAllocated(Generator::M_NUM_SAMPLES);

    juce::Array<var>* values = m_State.getProperty(Parameters::sampleProp).getArray();
    for(int i = 0; i < Generator::M_NUM_SAMPLES; i++)
        sample.insert(i, (*values)[i]);

    m_AudioContainerPtr->audio.addArray(sample);
}

void Planet::playSample(){
    Logger::writeToLog("Playing audio...");
    m_AudioContainerPtr->sampleIndex.clear();
    m_AudioContainerPtr->playAudio = true;
}

//--------------------------------------------------//
// Private methods.

bool Planet::hitTest(int x, int y){
    float a = pow((float)x - ((float)getDiameter() + (float)getClipBoundary()) / 2.0f, 2.0f);
    float b = pow((float)y - ((float)getDiameter() + (float)getClipBoundary()) / 2.0f, 2.0f);
    return sqrt(a + b) <= getDiameter() / 2;
}

void Planet::mouseDown(const MouseEvent& e){
    // Starts dragging component.
    m_Dragger.startDraggingComponent(this, e);
}

void Planet::mouseUp(const MouseEvent& e){
    if(e.mods.isLeftButtonDown()){
        // Generates new sample if double clicked with left mouse button.
        if(e.getNumberOfClicks() > 1){
            Logger::writeToLog("Generating sample...");

            // TODO:
            // NEED TO GENERATE NEW SAMPLE.

            Logger::writeToLog("Sample generated.");
        }
        
        // Plays sample if clicked once with left mouse button.
        else if(e.getNumberOfClicks() == 1 && e.mouseWasClicked()){
            addSample();
            playSample();
        }

        else if(e.mouseWasDraggedSinceMouseDown()){
            updateGraph();
        }
    }
    
    // Destroys planet if clicked with right mouse button.
    else if(e.mods.isRightButtonDown()){
        // Initializes planet destruction.
        m_Destroy.setValue(true);
        Logger::writeToLog("Set to destroy.");
    }
}

void Planet::mouseDrag(const MouseEvent& e){
    m_Dragger.dragComponent(this, e, nullptr);
    checkCollision();
    checkBounds();
    setPosXY(getX(), getY());
}

void Planet::mouseWheelMove(const MouseEvent& e, const MouseWheelDetails& w){    
    Logger::writeToLog("Wheel moved.");

    if(w.deltaY > 0.0f && getDiameter() < Variables::MAX_PLANET_SIZE){
        resizePlanet(getDiameter() + Variables::SIZE_MODIFIER);
    }
    else if(w.deltaY < 0.0f && getDiameter() > Variables::MIN_PLANET_SIZE){
        resizePlanet(getDiameter() - Variables::SIZE_MODIFIER);
    }
}

void Planet::visibilityChanged(){}

void Planet::checkCollision(){
    int centrePosX = getCentreX(this);
    int centrePosY = getCentreY(this);

    float distance, minDistance;

    // Check collision with sun.
    {
        int centreXSun = Variables::WINDOW_WIDTH / 2;
        int centreYSun = Variables::WINDOW_HEIGHT / 2;
        int sunDiameter = Variables::SUN_DIAMETER;

        distance = getDistance(centrePosX, centrePosY, centreXSun, centreYSun);
        minDistance = (sunDiameter + getDiameter()) / 2;

        if(distance <= minDistance){
            draw(getDiameter(), getPosX(), getPosY());
        }
    }

    Planet* planet;
    int centrePosX2, centrePosY2;

    for(int i = 0; i < m_PlanetsPtr->size(); i++){
        // Variable for ease of use.
        planet = (*m_PlanetsPtr)[i];

        // Avoid self collision testing.
        if(planet->getComponentID() != getComponentID()){
            centrePosX2 = getCentreX(planet);
            centrePosY2 = getCentreY(planet);

            distance = getDistance(centrePosX, centrePosY, centrePosX2, centrePosY2);
            minDistance = (planet->getDiameter() + getDiameter()) / 2;

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
    if(getX() + getDiameter() + (getClipBoundary() / 2) > getMapWidth())
        draw(getDiameter(), getMapWidth() - getDiameter() - (getClipBoundary() / 2), getY());

    // Check bottom boundary.
    if(getY() + getDiameter() + (getClipBoundary() / 2) > getMapHeight())
        draw(getDiameter(), getX(), getMapHeight() - getDiameter() - (getClipBoundary() / 2));
}
