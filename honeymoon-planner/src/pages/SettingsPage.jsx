import { useState } from 'react';
import { useHoneymoon } from '../context/HoneymoonContext';
import { Settings, Save, X, Plus, Trash2 } from 'lucide-react';

export default function SettingsPage() {
  const { tripInfo, setTripInfo } = useHoneymoon();
  const [formData, setFormData] = useState({ ...tripInfo });
  const [newDestination, setNewDestination] = useState('');
  const [saved, setSaved] = useState(false);

  const handleSave = (e) => {
    e.preventDefault();
    setTripInfo(formData);
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  const addDestination = () => {
    if (newDestination.trim() && !formData.destinations.includes(newDestination.trim())) {
      setFormData({
        ...formData,
        destinations: [...formData.destinations, newDestination.trim()]
      });
      setNewDestination('');
    }
  };

  const removeDestination = (city) => {
    setFormData({
      ...formData,
      destinations: formData.destinations.filter(d => d !== city)
    });
  };

  return (
    <div className="settings-page">
      <header className="page-header">
        <div className="header-content">
          <Settings size={32} />
          <div>
            <h1>Trip Settings</h1>
            <p>Edit your honeymoon details</p>
          </div>
        </div>
      </header>

      <form onSubmit={handleSave} className="settings-form">
        <div className="settings-card">
          <h2>Couple Name</h2>
          <div className="form-group">
            <label>How should we address you?</label>
            <input
              type="text"
              value={formData.couple}
              onChange={(e) => setFormData({ ...formData, couple: e.target.value })}
              placeholder="e.g., John & Jane"
            />
          </div>
        </div>

        <div className="settings-card">
          <h2>Travel Dates</h2>
          <div className="form-row">
            <div className="form-group">
              <label>Start Date</label>
              <input
                type="date"
                value={formData.startDate}
                onChange={(e) => setFormData({ ...formData, startDate: e.target.value })}
              />
            </div>
            <div className="form-group">
              <label>End Date</label>
              <input
                type="date"
                value={formData.endDate}
                onChange={(e) => setFormData({ ...formData, endDate: e.target.value })}
              />
            </div>
          </div>
        </div>

        <div className="settings-card">
          <h2>Destinations</h2>
          <div className="destinations-list">
            {formData.destinations.map(city => (
              <div key={city} className="destination-tag">
                <span>{city}</span>
                <button type="button" onClick={() => removeDestination(city)}>
                  <Trash2 size={14} />
                </button>
              </div>
            ))}
          </div>
          <div className="add-destination">
            <input
              type="text"
              value={newDestination}
              onChange={(e) => setNewDestination(e.target.value)}
              placeholder="Add a city..."
              onKeyPress={(e) => e.key === 'Enter' && (e.preventDefault(), addDestination())}
            />
            <button type="button" className="btn-secondary" onClick={addDestination}>
              <Plus size={16} /> Add
            </button>
          </div>
        </div>

        <div className="settings-actions">
          <button type="submit" className="btn-primary">
            <Save size={18} /> {saved ? 'Saved!' : 'Save Changes'}
          </button>
        </div>
      </form>
    </div>
  );
}
