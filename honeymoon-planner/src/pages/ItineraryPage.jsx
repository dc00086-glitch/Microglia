import { useState } from 'react';
import { useHoneymoon } from '../context/HoneymoonContext';
import {
  Calendar, Plus, Trash2, Edit2, Clock, MapPin, Save, X, ChevronDown, ChevronUp
} from 'lucide-react';
import { format, parseISO } from 'date-fns';

function ActivityItem({ activity, dayId, onEdit, onDelete }) {
  return (
    <div className="activity-item">
      <div className="activity-time">
        <Clock size={14} />
        <span>{activity.time}</span>
      </div>
      <div className="activity-content">
        <h4>{activity.title}</h4>
        {activity.notes && <p className="activity-notes">{activity.notes}</p>}
      </div>
      <div className="activity-actions">
        <button className="icon-btn" onClick={() => onEdit(activity)}>
          <Edit2 size={16} />
        </button>
        <button className="icon-btn delete" onClick={() => onDelete(dayId, activity.id)}>
          <Trash2 size={16} />
        </button>
      </div>
    </div>
  );
}

function DayCard({ day, onAddActivity, onEditActivity, onDeleteActivity, onDeleteDay }) {
  const [expanded, setExpanded] = useState(true);
  const [showAddForm, setShowAddForm] = useState(false);
  const [editingActivity, setEditingActivity] = useState(null);
  const [newActivity, setNewActivity] = useState({ time: '09:00', title: '', notes: '' });

  const handleAddActivity = (e) => {
    e.preventDefault();
    if (newActivity.title.trim()) {
      onAddActivity(day.id, newActivity);
      setNewActivity({ time: '09:00', title: '', notes: '' });
      setShowAddForm(false);
    }
  };

  const handleEditActivity = (activity) => {
    setEditingActivity({ ...activity });
    setShowAddForm(false);
  };

  const handleSaveEdit = (e) => {
    e.preventDefault();
    if (editingActivity.title.trim()) {
      onEditActivity(day.id, editingActivity.id, editingActivity);
      setEditingActivity(null);
    }
  };

  return (
    <div className="day-card">
      <div className="day-header" onClick={() => setExpanded(!expanded)}>
        <div className="day-info">
          <span className="day-number">Day {day.day}</span>
          <span className="day-date">{format(parseISO(day.date), 'EEEE, MMMM d')}</span>
          <div className="day-location">
            <MapPin size={14} />
            <span>{day.city}, {day.country}</span>
          </div>
        </div>
        <div className="day-header-actions">
          <span className="activity-count">{day.activities.length} activities</span>
          {expanded ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
        </div>
      </div>

      {expanded && (
        <div className="day-content">
          <div className="activities-list">
            {day.activities
              .sort((a, b) => a.time.localeCompare(b.time))
              .map(activity => (
                editingActivity?.id === activity.id ? (
                  <form key={activity.id} className="activity-form" onSubmit={handleSaveEdit}>
                    <input
                      type="time"
                      value={editingActivity.time}
                      onChange={(e) => setEditingActivity({ ...editingActivity, time: e.target.value })}
                    />
                    <input
                      type="text"
                      placeholder="Activity title"
                      value={editingActivity.title}
                      onChange={(e) => setEditingActivity({ ...editingActivity, title: e.target.value })}
                    />
                    <input
                      type="text"
                      placeholder="Notes (optional)"
                      value={editingActivity.notes}
                      onChange={(e) => setEditingActivity({ ...editingActivity, notes: e.target.value })}
                    />
                    <div className="form-actions">
                      <button type="submit" className="btn-save"><Save size={16} /> Save</button>
                      <button type="button" className="btn-cancel" onClick={() => setEditingActivity(null)}>
                        <X size={16} /> Cancel
                      </button>
                    </div>
                  </form>
                ) : (
                  <ActivityItem
                    key={activity.id}
                    activity={activity}
                    dayId={day.id}
                    onEdit={handleEditActivity}
                    onDelete={onDeleteActivity}
                  />
                )
              ))}
          </div>

          {showAddForm ? (
            <form className="activity-form" onSubmit={handleAddActivity}>
              <input
                type="time"
                value={newActivity.time}
                onChange={(e) => setNewActivity({ ...newActivity, time: e.target.value })}
              />
              <input
                type="text"
                placeholder="Activity title"
                value={newActivity.title}
                onChange={(e) => setNewActivity({ ...newActivity, title: e.target.value })}
                autoFocus
              />
              <input
                type="text"
                placeholder="Notes (optional)"
                value={newActivity.notes}
                onChange={(e) => setNewActivity({ ...newActivity, notes: e.target.value })}
              />
              <div className="form-actions">
                <button type="submit" className="btn-save"><Plus size={16} /> Add</button>
                <button type="button" className="btn-cancel" onClick={() => setShowAddForm(false)}>
                  <X size={16} /> Cancel
                </button>
              </div>
            </form>
          ) : (
            <button className="add-activity-btn" onClick={() => setShowAddForm(true)}>
              <Plus size={16} /> Add Activity
            </button>
          )}

          <button className="delete-day-btn" onClick={() => onDeleteDay(day.id)}>
            <Trash2 size={14} /> Remove Day
          </button>
        </div>
      )}
    </div>
  );
}

export default function ItineraryPage({ filterCity, clearFilter }) {
  const { itinerary, tripInfo, addDay, deleteDay, addActivity, updateActivity, deleteActivity } = useHoneymoon();
  const [showNewDayForm, setShowNewDayForm] = useState(false);
  const [newDay, setNewDay] = useState({
    day: itinerary.length + 1,
    date: '',
    city: filterCity || '',
    country: '',
    activities: []
  });

  const handleAddDay = (e) => {
    e.preventDefault();
    if (newDay.date && newDay.city && newDay.country) {
      addDay(newDay);
      setNewDay({
        day: itinerary.length + 2,
        date: '',
        city: filterCity || '',
        country: '',
        activities: []
      });
      setShowNewDayForm(false);
    }
  };

  const sortedItinerary = [...itinerary]
    .filter(day => !filterCity || day.city === filterCity)
    .sort((a, b) => a.day - b.day);

  const allCities = [...new Set(itinerary.map(day => day.city))];

  return (
    <div className="itinerary-page">
      <header className="page-header">
        <div className="header-content">
          <Calendar size={32} />
          <div>
            <h1>{filterCity ? `${filterCity} Itinerary` : 'Trip Itinerary'}</h1>
            <p>{filterCity ? `Your plans for ${filterCity}` : 'Plan your perfect days across Europe'}</p>
          </div>
        </div>
        <div className="header-actions">
          {filterCity && (
            <button className="btn-secondary" onClick={clearFilter}>
              <X size={18} /> Show All
            </button>
          )}
          <button className="btn-primary" onClick={() => setShowNewDayForm(true)}>
            <Plus size={18} /> Add Day
          </button>
        </div>
      </header>

      {!filterCity && allCities.length > 0 && (
        <div className="city-filter-bar">
          <span>Filter by city:</span>
          {tripInfo.destinations.map(city => {
            const count = itinerary.filter(day => day.city === city).length;
            return (
              <button
                key={city}
                className="filter-btn"
                onClick={() => clearFilter(city)}
              >
                {city} ({count})
              </button>
            );
          })}
        </div>
      )}

      {showNewDayForm && (
        <div className="modal-overlay" onClick={() => setShowNewDayForm(false)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <h2>Add New Day</h2>
            <form onSubmit={handleAddDay}>
              <div className="form-group">
                <label>Day Number</label>
                <input
                  type="number"
                  value={newDay.day}
                  onChange={(e) => setNewDay({ ...newDay, day: parseInt(e.target.value) })}
                  min="1"
                />
              </div>
              <div className="form-group">
                <label>Date</label>
                <input
                  type="date"
                  value={newDay.date}
                  onChange={(e) => setNewDay({ ...newDay, date: e.target.value })}
                  required
                />
              </div>
              <div className="form-group">
                <label>City</label>
                <input
                  type="text"
                  placeholder="e.g., Rome"
                  value={newDay.city}
                  onChange={(e) => setNewDay({ ...newDay, city: e.target.value })}
                  required
                />
              </div>
              <div className="form-group">
                <label>Country</label>
                <input
                  type="text"
                  placeholder="e.g., Italy"
                  value={newDay.country}
                  onChange={(e) => setNewDay({ ...newDay, country: e.target.value })}
                  required
                />
              </div>
              <div className="modal-actions">
                <button type="submit" className="btn-primary">
                  <Plus size={16} /> Add Day
                </button>
                <button type="button" className="btn-secondary" onClick={() => setShowNewDayForm(false)}>
                  Cancel
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      <div className="itinerary-list">
        {sortedItinerary.length === 0 ? (
          <div className="empty-state">
            <Calendar size={48} />
            <h3>No days planned yet</h3>
            <p>Start planning your honeymoon adventure!</p>
            <button className="btn-primary" onClick={() => setShowNewDayForm(true)}>
              <Plus size={18} /> Add Your First Day
            </button>
          </div>
        ) : (
          sortedItinerary.map(day => (
            <DayCard
              key={day.id}
              day={day}
              onAddActivity={addActivity}
              onEditActivity={updateActivity}
              onDeleteActivity={deleteActivity}
              onDeleteDay={deleteDay}
            />
          ))
        )}
      </div>
    </div>
  );
}
