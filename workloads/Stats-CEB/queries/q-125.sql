SELECT COUNT(*)
FROM postHistory AS ph,
  votes AS v,
  users AS u,
  badges AS b
WHERE u.Id = b.UserId
  AND u.Id = ph.UserId
  AND u.Id = v.UserId
  AND ph.PostHistoryTypeId = 1
  AND v.CreationDate <= CAST('2014-09-12 00:00:00' AS timestamp)
  AND u.Reputation <= 126
  AND u.Views <= 11
  AND u.CreationDate >= CAST('2010-08-02 16:17:58' AS timestamp)
  AND u.CreationDate <= CAST('2014-09-12 00:16:30' AS timestamp)
  AND b.Date <= CAST('2014-09-03 16:13:12' AS timestamp);
