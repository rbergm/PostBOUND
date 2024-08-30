SELECT COUNT(*)
FROM comments AS c, postHistory AS ph, users AS u
WHERE u.Id = c.UserId
  AND c.UserId = ph.UserId
  AND u.Reputation >= 1
  AND u.Reputation <= 487
  AND u.UpVotes <= 27
  AND u.CreationDate >= CAST('2010-10-22 22:40:35' AS timestamp)
  AND u.CreationDate <= CAST('2014-09-10 17:01:31' AS timestamp);
