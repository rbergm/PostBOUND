SELECT COUNT(*)
FROM comments AS c,
  postHistory AS ph,
  badges AS b,
  users AS u
WHERE u.Id = c.UserId
  AND u.Id = ph.UserId
  AND u.Id = b.UserId
  AND c.Score = 0
  AND c.CreationDate >= CAST('2010-09-05 16:04:35' AS timestamp)
  AND c.CreationDate <= CAST('2014-09-11 04:35:36' AS timestamp)
  AND ph.PostHistoryTypeId = 1
  AND ph.CreationDate >= CAST('2010-07-26 20:01:58' AS timestamp)
  AND ph.CreationDate <= CAST('2014-09-13 17:29:23' AS timestamp)
  AND b.Date <= CAST('2014-09-04 08:54:56' AS timestamp);
